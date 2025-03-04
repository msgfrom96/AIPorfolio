#!/usr/bin/env python3
"""
Urban Dish Analyzer
-------------------
Analyzes restaurant reviews for a given city by extracting topics, sentiment, and cuisine.
Clusters restaurants using a hybrid approach (grid-based clustering merged by business profiles),
and visualizes the results on an interactive map with popups showing sentiment, topics, similar
restaurants, and city analytics. Includes fallback if grid-based clusters are 0, plus SettingWithCopy fixes.
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import traceback
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from folium.plugins import MarkerCluster, Fullscreen, LocateControl, MeasureControl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(1)
sys.setrecursionlimit(5000)

# --- Configuration ---
@dataclass
class UDAConfig:
    city: str = "tucson"
    cache_dir: str = field(default_factory=lambda: ".cache/tucson")
    data_dir: str = "data"
    sample_limit: int = 100
    chunksize: int = 1000
    max_reviews: int = 1000
    cache_ttl: int = 3600
    cache_version: str = "v3.1"
    zero_shot_model: str = "valhalla/distilbart-mnli-12-3"
    topic_confidence_threshold: float = 0.3
    aspect_confidence_threshold: float = 0.5
    topics: List[str] = field(default_factory=lambda: [
        "food quality", "service quality", "ambiance", "price", "location",
        "dietary accommodations", "taste characteristics", "cuisine"
    ])
    topic_colors: Dict[str, str] = field(default_factory=lambda: {
        "food quality": "#1f77b4",
        "service quality": "#ff7f0e",
        "ambiance": "#2ca02c",
        "price": "#d62728",
        "location": "#9467bd",
        "dietary accommodations": "#8c564b",
        "taste characteristics": "#e377c2",
        "cuisine": "#9b59b6"
    })
    cluster_params: Dict[str, Any] = field(default_factory=lambda: {
        "cell_size": 0.05,
        "min_count": 5,   # can reduce to 2 or 1 if necessary
        "max_cluster_radius": 5.0
    })
    map_width: str = "100%"
    map_height: str = "750px"
    positive_sentiment_threshold: float = 0.6
    negative_sentiment_threshold: float = 0.4
    max_categories: int = 5
    max_keywords: int = 7

    def get_topic_color(self, topic: str) -> str:
        return self.topic_colors.get(topic, "#000000")

# --- Logging & Utils ---
def setup_logging(level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/uda.log"), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("UrbanDishAnalyzer")

logger = setup_logging()

def calculate_data_hash(df: pd.DataFrame) -> str:
    try:
        dfc = df.copy()
        for col in dfc.columns:
            if dfc[col].dtype == 'object':
                dfc[col] = dfc[col].apply(lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list)) else x)
        dfc = dfc.replace([float('inf'), float('-inf')], 'inf')
        return hashlib.md5(pd.util.hash_pandas_object(dfc, index=True).values.tobytes()).hexdigest()
    except Exception as e:
        logger.error(f"Data hash error: {e}")
        return "default_hash"

def download_nltk_resources():
    for r in ['wordnet', 'punkt', 'stopwords']:
        try:
            nltk.data.find(f'corpora/{r}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {r}")
            nltk.download(r)

# --- Cache ---
class Cache:
    def __init__(self, config: UDAConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.logger = logging.getLogger("UDA.Cache")
    
    def get_path(self, component: str, key: str, data_hash: Optional[str] = None) -> Path:
        fname = f"{self.config.cache_version}_{component}_{key}" + (f"_{data_hash}" if data_hash else "")
        return self.cache_dir / (fname + ".parquet")
    
    def get(self, component: str, key: str, data_hash: Optional[str] = None) -> Any:
        ckey = f"{component}_{key}" + (f"_{data_hash}" if data_hash else "")
        if ckey in self.cache:
            return self.cache[ckey]
        path = self.get_path(component, key, data_hash)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                self.cache[ckey] = df
                return df
            except Exception as e:
                self.logger.warning(f"Cache load error {path}: {e}")
        return None
    
    def set(self, component: str, key: str, value: Any, data_hash: Optional[str] = None) -> None:
        ckey = f"{component}_{key}" + (f"_{data_hash}" if data_hash else "")
        path = self.get_path(component, key, data_hash)
        try:
            pd.DataFrame(value).to_parquet(path)
            self.cache[ckey] = value
        except Exception as e:
            self.logger.warning(f"Cache save error {path}: {e}")

# --- DataLoader ---
class DataLoader:
    def __init__(self, config: UDAConfig, cache: Cache):
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger("UDA.DataLoader")
    
    def get_cache_key(self, base: str, **kwargs) -> str:
        return "_".join([base] + [f"{k}_{v}" for k, v in kwargs.items() if v is not None])
    
    def load_business_data(self) -> pd.DataFrame:
        key = self.get_cache_key("business", city=self.config.city)
        cached = self.cache.get("DataLoader", key)
        if cached is not None:
            logger.info("Using cached business data")
            return cached
        
        # Attempt to find a business JSON file
        business_file = None
        for fname in ["business.json", "yelp_academic_dataset_business.json"]:
            fpath = os.path.join(self.config.data_dir, fname)
            if os.path.exists(fpath):
                business_file = fpath
                break
        if not business_file:
            raise FileNotFoundError("Business data file not found.")
        
        logger.info(f"Reading business data from {business_file}")
        chunks = []
        for chunk in pd.read_json(business_file, lines=True, chunksize=self.config.chunksize):
            # Filter for city and restaurants
            filtered = chunk[
                (chunk["city"].str.lower() == self.config.city.lower()) &
                (chunk["categories"].str.contains("Restaurant", case=False, na=False))
            ]
            # Clean categories
            filtered.loc[:, "categories"] = filtered["categories"].apply(
                lambda x: ",".join(c.strip() for c in x.split(",") 
                                   if c.strip().lower() not in ["restaurant","restaurants","food"])
                if isinstance(x, str) else x
            )
            chunks.append(filtered)
            if self.config.sample_limit and sum(len(c) for c in chunks) >= self.config.sample_limit:
                break
        if not chunks:
            raise ValueError("No restaurant businesses found for this city.")
        
        businesses = pd.concat(chunks, ignore_index=True)
        for col in ["business_id", "name", "address", "city", "state", "postal_code",
                    "latitude", "longitude", "stars", "review_count", "categories"]:
            if col not in businesses.columns:
                businesses[col] = None
        
        logger.info(f"Loaded {len(businesses)} businesses")
        self.cache.set("DataLoader", key, businesses)
        return businesses
    
    def _safe_read_json_lines(self, file_path: str, chunksize: int):
        valid = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    valid.append(json.loads(line))
                    if len(valid) >= chunksize:
                        yield pd.DataFrame(valid)
                        valid = []
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error reading review line: {e}")
            if valid:
                yield pd.DataFrame(valid)

    def load_reviews(self, business_ids: Set[str]) -> pd.DataFrame:
        key = self.get_cache_key("reviews", city=self.config.city, 
                                 num_businesses=len(business_ids), 
                                 max_reviews=self.config.max_reviews)
        cached = self.cache.get("DataLoader", key)
        if cached is not None:
            logger.info("Using cached review data")
            reviews = cached[cached["business_id"].isin(business_ids)]
            if self.config.max_reviews and len(reviews) > self.config.max_reviews:
                reviews = reviews.sample(n=self.config.max_reviews, random_state=42)
            return reviews
        
        review_file = None
        for fname in ["review.json", "yelp_academic_dataset_review.json"]:
            fpath = os.path.join(self.config.data_dir, fname)
            if os.path.exists(fpath):
                review_file = fpath
                break
        
        if not review_file:
            self.logger.error("Review data file not found.")
            return pd.DataFrame()
        
        logger.info(f"Loading reviews for {len(business_ids)} businesses from {review_file}")
        all_reviews = []
        total_read = 0
        max_per_business = max(5, int(self.config.max_reviews / max(1, len(business_ids))))
        counters = {bid: 0 for bid in business_ids}
        
        for chunk in self._safe_read_json_lines(review_file, self.config.chunksize):
            # Filter chunk
            df_chunk = chunk[chunk["business_id"].isin(business_ids)]
            if df_chunk.empty:
                continue
            keep_idxs = []
            for idx, row in df_chunk.iterrows():
                bid = row["business_id"]
                if counters[bid] < max_per_business:
                    keep_idxs.append(idx)
                    counters[bid] += 1
            df_chunk = df_chunk.loc[keep_idxs]
            if not df_chunk.empty:
                all_reviews.append(df_chunk)
                total_read += len(df_chunk)
            if self.config.max_reviews and total_read >= self.config.max_reviews:
                self.logger.info(f"Reached max review limit ({self.config.max_reviews})")
                break
        
        if not all_reviews:
            logger.warning("No reviews found matching these businesses.")
            return pd.DataFrame()
        
        reviews_df = pd.concat(all_reviews, ignore_index=True)
        if self.config.max_reviews and len(reviews_df) > self.config.max_reviews:
            reviews_df = reviews_df.sample(n=self.config.max_reviews, random_state=42)
        
        self.cache.set("DataLoader", key, reviews_df)
        logger.info(f"Loaded {len(reviews_df)} reviews")
        return reviews_df

# --- AspectExtractor ---
class AspectExtractor:
    def __init__(self, config: UDAConfig, cache: Cache):
        # Initialize spaCy
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except ImportError:
            raise RuntimeError("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        
        self.allowed_pos = {"NOUN", "ADJ", "VERB"}
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger("UDA.AspectExtractor")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.max_length = 512

    def load_zero_shot_model(self):
        model_name = self.config.zero_shot_model
        self.logger.info(f"Loading zero-shot classification model on {self.device}: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = pipeline("zero-shot-classification",
                                       model=AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device),
                                       tokenizer=self.tokenizer,
                                       device=0 if (self.device == "cuda") else -1)
        except Exception as e:
            self.logger.error(f"Zero-shot model error: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def load_sentiment_model(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.logger.info(f"Loading sentiment analysis model on {self.device}: {model_name}")
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline("sentiment-analysis",
                                               model=model_name,
                                               tokenizer=tok,
                                               device=0 if (self.device == "cuda") else -1,
                                               truncation=True, max_length=self.max_length)
        except Exception as e:
            self.logger.error(f"Sentiment model error: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def preprocess_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Batch processing reviews (topics + sentiment)...")
        df = reviews_df.copy()
        texts = df["text"].astype(str).tolist()
        
        # Topics
        self.logger.info("Extracting topics in batch...")
        topics_res = self.classifier(
            texts,
            candidate_labels=self.config.topics,
            multi_label=True,
            truncation=True,
            max_length=self.max_length,
            batch_size=16
        )
        df["topic_scores"] = [dict(zip(r["labels"], r["scores"])) for r in topics_res]

        # Sentiment
        self.logger.info("Analyzing sentiment in batch...")
        sentiments = self.sentiment_pipeline(texts, batch_size=16)
        # Possibly adjust "neutral" based on star rating, if needed
        adjusted = []
        for i, sres in enumerate(sentiments):
            label = sres["label"].lower()  # "POSITIVE", "NEUTRAL", "NEGATIVE"
            if label == "neutral" and sres["score"] < 0.7:
                # Attempt star-based override
                stars = df.iloc[i]["stars"]
                if pd.notna(stars):
                    if stars < 3:
                        label = "negative"
                    elif stars > 3:
                        label = "positive"
                    else:
                        label = "neutral"
            adjusted.append(label)
        df["sentiment_score"] = adjusted

        return df

    def classify_reviews(self, reviews_df: pd.DataFrame, businesses_df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess_reviews(reviews_df)
        self.logger.info("Aggregating results per business...")
        results = {}
        stops = set(stopwords.words("english"))

        for bid, group in df.groupby("business_id"):
            # Sentiment calculation with star fallback
            total = len(group)
            pos_ct = (group["sentiment_score"] == "positive").sum()
            neg_ct = (group["sentiment_score"] == "negative").sum()
            neu_ct = total - pos_ct - neg_ct
            
            # Get star rating from BUSINESS data (not reviews)
            stars = businesses_df[businesses_df["business_id"] == bid]["stars"].values[0]
            if total == 0 or pd.isna(stars):  # Fallback to stars if no reviews
                pos = neu = neg = 0.0
                if stars >= 4: pos = 1.0
                elif stars <= 2: neg = 1.0
                else: neu = 1.0
            else:
                pos = pos_ct / total
                neu = neu_ct / total 
                neg = neg_ct / total

            results[bid] = {
                "sentiment": {"positive": pos, "neutral": neu, "negative": neg},
                "topics": group["topic_scores"].iloc[0],
                "keywords": self._filter_keywords(group["text"].iloc[0])
            }
        
        # Merge results back into businesses dataframe
        businesses_df = businesses_df.merge(
            pd.DataFrame.from_dict(results, orient="index"),
            left_on="business_id",
            right_index=True
        )
        return businesses_df

    def _filter_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords using spaCy"""
        doc = self.nlp(text.lower())
        return [
            token.lemma_ for token in doc
            if (token.pos_ in self.allowed_pos and
                not token.is_stop and
                not token.is_punct and
                len(token.text) > 2)
        ]

    def _process_zero_shot(self, text: str) -> Dict[str, float]:
        """Process zero-shot results with calibration"""
        result = self.classifier(
            text,
            candidate_labels=self.config.topics,
            multi_label=True
        )
        
        # Calibrate scores using temperature scaling
        calibrated = torch.softmax(
            torch.tensor(result["scores"]) / 0.7,  # T=0.7 for less peaky
            dim=0
        ).tolist()

        # Apply confidence threshold and normalize
        topics = {
            label: score 
            for label, score in zip(result["labels"], calibrated)
            if score >= self.config.topic_confidence_threshold
        }
        
        # Normalize to sum=1
        total = sum(topics.values())
        return {k: v/total for k, v in topics.items()} if total > 0 else {}

# --- GridAnalyzer ---
class GridAnalyzer:
    def __init__(self, config: UDAConfig, cache: Cache):
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger("UDA.GridAnalyzer")
        self.business_data = {}
        self.grid = {}
        self.tfidf_vectorizer = None
    
    @property
    def clusters(self):
        return self.grid

    def load_business_data(self, businesses_df: pd.DataFrame, _reviews_df: pd.DataFrame):
        self.business_data = {
            row["business_id"]: {
                "business_id": row["business_id"],
                "name": row["name"],
                "address": row.get("address", ""),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "stars": row["stars"],
                "review_count": row["review_count"],
                "categories": row.get("categories", ""),
                "topics": row.get("topics", {}),
                "sentiment": row.get("sentiment", {"positive":0, "neutral":0, "negative":0}),
                "keywords": row.get("keywords", []),
            }
            for _, row in businesses_df.iterrows()
        }
        self.logger.info(f"Loaded data for {len(self.business_data)} businesses")
        self._init_tfidf()

    def _init_tfidf(self):
        texts = []
        for bdata in self.business_data.values():
            if bdata.get("keywords"):
                texts.append(" ".join(bdata["keywords"]))
            else:
                texts.append("no keywords")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5, stop_words="english")
        self.tfidf_vectorizer.fit(texts)

    def _create_feature_vector(self, bid: str) -> np.ndarray:
        b = self.business_data[bid]
        coords = np.array([b["latitude"], b["longitude"]])
        rating = np.array([b["stars"], b["review_count"]/(self.config.max_reviews or 1)])
        # topic vector
        tvec = np.array([b["topics"].get(t, 0) for t in self.config.topics])
        # keyword vector
        text = " ".join(b["keywords"]) if b.get("keywords") else "no keywords"
        try:
            kw = self.tfidf_vectorizer.transform([text]).toarray()[0]
        except:
            kw = np.zeros(5)
        return np.concatenate([coords, rating, tvec, kw])

    def get_dominant_topic(self, bids: List[str]) -> str:
        totals = defaultdict(float)
        total_reviews = 0
        for bid in bids:
            b = self.business_data[bid]
            for t, s in b["topics"].items():
                totals[t] += s
            total_reviews += b["review_count"] or 0
        if not totals:
            return "Unknown"
        # average
        div = total_reviews if total_reviews else len(bids)
        avgs = {k: v/div for k,v in totals.items()}
        return max(avgs.items(), key=lambda x: x[1])[0]

    def _grid_based_clustering(self, businesses_df: pd.DataFrame) -> None:
        self.grid = {}
        if not self.business_data:
            self.logger.warning("No business data for clustering.")
            return
        
        min_lat = min(b["latitude"] for b in self.business_data.values())
        max_lat = max(b["latitude"] for b in self.business_data.values())
        min_lon = min(b["longitude"] for b in self.business_data.values())
        max_lon = max(b["longitude"] for b in self.business_data.values())
        width = int((max_lon - min_lon) / self.config.cluster_params["cell_size"]) + 1
        height = int((max_lat - min_lat) / self.config.cluster_params["cell_size"]) + 1
        self.logger.info(f"Grid dimensions: width={width}, height={height}")
        valid = 0

        for cx in range(width):
            for cy in range(height):
                cell_lat = min_lat + cy*self.config.cluster_params["cell_size"]
                cell_lon = min_lon + cx*self.config.cluster_params["cell_size"]
                # find businesses in the cell
                bids = [bid for bid, val in self.business_data.items()
                        if cell_lat <= val["latitude"] < cell_lat + self.config.cluster_params["cell_size"]
                        and cell_lon <= val["longitude"] < cell_lon + self.config.cluster_params["cell_size"]]
                if len(bids) >= self.config.cluster_params["min_count"]:
                    center = {
                        "lat": cell_lat + self.config.cluster_params["cell_size"]/2,
                        "lon": cell_lon + self.config.cluster_params["cell_size"]/2
                    }
                    dom_topic = self.get_dominant_topic(bids)
                    # attempt hull if >=3
                    hull = []
                    if len(bids) >= 3:
                        pts = np.array([[self.business_data[bid]["latitude"], self.business_data[bid]["longitude"]] for bid in bids])
                        if len(np.unique(pts, axis=0)) >= 3:
                            try:
                                hull = pts[ConvexHull(pts).vertices].tolist()
                            except Exception as e:
                                self.logger.warning(f"Hull error ({cx},{cy}): {e}")
                    self.grid[f"{cx}_{cy}"] = {
                        "center": center,
                        "business_ids": bids,
                        "dominant_topic": dom_topic,
                        "hull_vertices": hull
                    }
                    valid += 1
        
        if valid == 0:
            self.logger.warning("No valid grid clusters were created; using fallback: each business is its own cluster.")
            for bid, data in self.business_data.items():
                self.grid[bid] = {
                    "center": {"lat": data["latitude"], "lon": data["longitude"]},
                    "business_ids": [bid],
                    "dominant_topic": self.get_dominant_topic([bid]),
                    "hull_vertices": []
                }
        self.logger.info(f"Created {len(self.grid)} grid clusters (fallback included if needed)")

    def calculate_clusters(self, businesses_df: pd.DataFrame, reviews_df: pd.DataFrame):
        # optionally compute data_hash
        dh = calculate_data_hash(businesses_df) + calculate_data_hash(reviews_df.head(min(len(reviews_df),100)))
        cached = self.cache.get("GridAnalyzer","clusters",dh)
        if cached is not None:
            self.grid = cached
            return
        self.logger.info("Calculating grid clusters...")
        self._grid_based_clustering(businesses_df)
        self.cache.set("GridAnalyzer","clusters",self.grid,dh)

    def hybrid_clusters(self, eps: float = 0.05, min_samples: int = 1):
        self.logger.info("Running hybrid clustering on grid centroids...")
        keys = list(self.grid.keys())
        centroids = []
        for k in keys:
            bids = self.grid[k]["business_ids"]
            if not bids:
                continue
            vecs = np.array([self._create_feature_vector(bid) for bid in bids])
            ctd = np.mean(vecs, axis=0)
            centroids.append(ctd)
        centroids = np.array(centroids)
        if len(centroids) < 1:
            self.logger.warning("No centroids available; skipping hybrid step.")
            return
        
        scaled = StandardScaler().fit_transform(centroids)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(scaled)
        labels = db.labels_
        self.logger.info(f"DBSCAN produced {len(set(labels)) - (1 if -1 in labels else 0)} merged clusters")

        merged = {}
        for label in set(labels):
            if label == -1:
                # keep them separate
                for k,lbl in zip(keys, labels):
                    if lbl == -1:
                        merged[k] = self.grid[k]
                continue
            idxs = [i for i,lbl in enumerate(labels) if lbl == label]
            mkey = "_".join([keys[i] for i in idxs])
            all_bids = []
            centers = []
            hulls = []
            topics_list = []
            for i in idxs:
                cluster_key = keys[i]
                all_bids.extend(self.grid[cluster_key]["business_ids"])
                c = self.grid[cluster_key]["center"]
                centers.append(np.array([c["lat"],c["lon"]]))
                if self.grid[cluster_key]["hull_vertices"]:
                    hulls.extend(self.grid[cluster_key]["hull_vertices"])
                topics_list.append(self.get_dominant_topic(self.grid[cluster_key]["business_ids"]))
            new_center = {"lat":float(np.mean([x[0] for x in centers])),
                          "lon":float(np.mean([x[1] for x in centers]))}
            # pick the topic that appears most among subclusters
            dt = Counter(topics_list).most_common(1)[0][0] if topics_list else "Unknown"
            merged_hull = []
            if len(hulls) >= 3:
                try:
                    arr = np.array(hulls)
                    if len(np.unique(arr, axis=0)) >= 3:
                        merged_hull = arr[ConvexHull(arr).vertices].tolist()
                except Exception as e:
                    self.logger.warning(f"Merged hull error: {e}")
            merged[mkey] = {
                "center": new_center,
                "business_ids": list(set(all_bids)),
                "dominant_topic": dt,
                "hull_vertices": merged_hull
            }
        
        self.grid = merged
        self.logger.info(f"Hybrid clustering complete. Final clusters: {len(self.grid)}")

    def _calculate_business_sentiment(self, reviews: pd.DataFrame) -> Dict[str, float]:
        """Robust sentiment calculation with threshold checks"""
        if reviews.empty:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
        # Use model if available
        if "sentiment" in reviews.columns:
            pos = reviews["sentiment"].apply(lambda x: x.get("positive", 0)).mean()
            neu = reviews["sentiment"].apply(lambda x: x.get("neutral", 0)).mean()
            neg = reviews["sentiment"].apply(lambda x: x.get("negative", 0)).mean()
        else:  # Fallback to star-based
            avg_stars = reviews["stars"].mean()
            pos = max(0, min(1, (avg_stars - 1.5)/3.5))  # 1.5★=0%, 5★=100%
            neg = max(0, min(1, (5 - avg_stars)/3.5))    # 5★=0%, 1.5★=100%
            neu = 1 - (pos + neg)

        # Apply thresholds
        pos = pos if pos >= self.config.positive_sentiment_threshold else 0
        neg = neg if neg >= self.config.negative_sentiment_threshold else 0
        neu = 1 - (pos + neg)  # Recalculate after thresholding
        
        return {
            "positive": pos * 100,
            "neutral": neu * 100,
            "negative": neg * 100
        }

# --- PopupGenerator ---
class PopupGenerator:
    def __init__(self, config: UDAConfig, business_data: Dict[str, Dict]):
        self.config = config
        self.business_data = business_data
        self.logger = logging.getLogger("UDA.PopupGenerator")
        self._init_templates()

    def _init_templates(self):
        self.business_template = """<div class="popup-container">
<h3>{name}</h3>
<div class="rating">⭐ {stars} ({review_count} reviews)</div>
<div class="address">📌 {address}</div>
{categories_html}
<div class="section">
    <h4>📊 Topics</h4>
    {topics_html}
</div>
<div class="section">
    <h4>🔑 Keywords</h4>
    <div class="keywords">{keywords_html}</div>
</div>
<div class="section">
    <h4>🍴 Similar Restaurants</h4>
    {similar_html}
</div>
</div>"""

        self.cluster_template = """<div class="popup-container">
<h3>🗂 Cluster: {cluster_id}</h3>
<div class="cluster-meta">🍽️ {business_count} restaurants | 🌟 Avg rating: {avg_stars:.1f}</div>
<div class="section">
    <h4>🏷 Dominant Topic</h4>
    <div class="dominant-topic" style="background:{topic_color};color:white;padding:2px 5px;border-radius:3px">{dominant_topic}</div>
</div>
<div class="section">
    <h4>🔍 Top Keywords</h4>
    <div class="keywords">{keywords_html}</div>
</div>
<div class="section">
    <h4>📍 Top Restaurants</h4>
    {restaurant_list}
</div>
</div>"""

        self.popup_css = """
<style>
.popup-container { max-width: 350px; padding: 10px; font-family: Arial, sans-serif; }
.rating, .address { margin-bottom: 8px; color: #666; }
.section { margin: 10px 0; border-top: 1px solid #eee; padding-top: 10px; }
.topic-bar { height: 20px; background: #f0f0f0; margin: 5px 0; border-radius: 3px; overflow: hidden; }
.topic-fill { height: 100%; padding-left: 5px; color: white; line-height: 20px; font-size: 0.9em; }
.keyword { display: inline-block; background: #e0e0e0; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em; }
.restaurant-list { max-height: 150px; overflow-y: auto; }
</style>
"""

    def _compute_similar_restaurants(self, bid: str) -> List[str]:
        similarities = []
        current = self.business_data.get(bid, {})
        for obid, data in self.business_data.items():
            if obid == bid:
                continue
            sim = sum(min(current.get("topics",{}).get(t,0), data.get("topics",{}).get(t,0)) for t in set(current.get("topics",{})) & set(data.get("topics",{})))
            similarities.append((obid, data.get("name","Unknown"), sim))
        return [s[1] for s in sorted(similarities, key=lambda x: x[2], reverse=True)[:3] if s[2]>0]

    def create_business_popup(self, bid: str, data: Dict, grid_analyzer=None) -> str:
        try:
            # Star display with emojis
            stars_val = float(data.get("stars",0))
            review_count = int(data.get("review_count",0))
            stars_display = "🌟" * int(round(stars_val))
            
            # Categories with emoji
            cat = data.get("categories","")
            categories_html = ""
            if cat:
                categories_html = f'<div class="categories">📌 {cat}</div>'

            # Topic progress bars
            topics = data.get("topics",{})
            topics_html = ""
            if topics:
                for t, score in topics.items():
                    color = self.config.get_topic_color(t)
                    width = min(100, max(5, score * 100))
                    topics_html += f'''
                    <div class="topic-bar">
                        <div class="topic-fill" style="width:{width}%;background:{color}">
                            {t} ({score:.0%})
                        </div>
                    </div>'''

            # Keyword bubbles
            kws = data.get("keywords",[])
            keywords_html = " ".join([f'<span class="keyword">{kw}</span>' for kw in kws[:self.config.max_keywords]])

            # Similar restaurants
            simlist = self._compute_similar_restaurants(bid)
            similar_html = "<ul>" + "".join([f'<li>🍴 {name}</li>' for name in simlist]) + "</ul>"

            return self.popup_css + self.business_template.format(
                name=data.get("name","Unnamed"),
                stars=stars_display,
                review_count=review_count,
                address=data.get("address","No address"),
                categories_html=categories_html,
                topics_html=topics_html,
                keywords_html=keywords_html,
                similar_html=similar_html
            )
        except Exception as e:
            self.logger.error(f"Business popup error: {e}")
            return f"<div>Error: {e}</div>"
    
    def create_cluster_popup(self, cid: str, cluster: Dict, grid_analyzer) -> str:
        try:
            bids = cluster.get("business_ids",[])
            if len(bids) < self.config.cluster_params["min_count"]:
                return "<div>Cluster too small</div>"
            
            # Cluster statistics
            avg_stars = np.mean([grid_analyzer.business_data[b].get("stars",0) for b in bids])
            dt = cluster.get("dominant_topic","Unknown")
            color = self.config.get_topic_color(dt)
            
            # Keywords
            all_kw = [kw for b in bids for kw in grid_analyzer.business_data[b].get("keywords",[])]
            top_kw = [f'<span class="keyword">{w}</span>' for w,_ in Counter(all_kw).most_common(5)]
            
            # Top restaurants
            top_restaurants = sorted(
                [(grid_analyzer.business_data[b]["name"], grid_analyzer.business_data[b]["stars"]) 
                 for b in bids],
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            restaurant_list = "<ul>" + "".join(
                [f'<li>🌟 {name} ({stars:.1f})</li>' for name, stars in top_restaurants]
            ) + "</ul>"

            return self.popup_css + self.cluster_template.format(
                cluster_id=cid,
                business_count=len(bids),
                avg_stars=avg_stars,
                dominant_topic=dt,
                topic_color=color,
                keywords_html=" ".join(top_kw),
                restaurant_list=restaurant_list
            )
        except Exception as e:
            self.logger.error(f"Cluster popup error: {e}")
            return f"<div>Error: {e}</div>"

# --- MapVisualizer ---
class MapVisualizer:
    def __init__(self, config: UDAConfig, business_data: Dict[str, Dict]):
        self.config = config
        self.logger = logging.getLogger("UDA.MapVisualizer")
        self.popup_generator = PopupGenerator(config,business_data)
        self.map = folium.Map(location=[0,0],zoom_start=2)

    def add_analytics_panel(self, analytics: Dict[str,Any]) -> None:
        html = f"""
<div id="analytics-panel" style="position: fixed; bottom: 20px; left: 10px; z-index: 1000;
     background: white; padding: 15px; border-radius: 8px; border: 1px solid #ccc; width: 320px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
    <h3 style="margin-top:0;color:#2c3e50;">🏙️ {self.config.city.title()} Analytics</h3>
    
    <div class="analytics-section">
        <h4>📢 Most Discussed Topics</h4>
        {"".join([f'''
        <div class="topic-bar" style="margin:5px 0;">
            <div class="topic-fill" style="width:{min(100, t['count']*10)}%;background:{self.config.get_topic_color(t['topic'])}">
                {t['topic']} ({t['count']})
            </div>
        </div>''' for t in analytics.get("most_discussed_topics",[])[:3]])}
    </div>
    
    <div class="analytics-section">
        <h4>🏆 Top Rated</h4>
        <ul style="list-style:none;padding-left:0;">
        {"".join([f'<li>⭐ {r["name"]} ({r["stars"]:.1f})</li>' for r in analytics.get("top_rated",[])[:3]])}
        </ul>
    </div>
    
    <div class="analytics-section">
        <h4>📉 Common Complaints</h4>
        {" ".join([f'<span class="keyword" style="background:#ffebee;color:#c62828;">{c["complaint"]}</span>' 
                  for c in analytics.get("key_complaints",[])[:3]])}
    </div>
</div>
<style>
.analytics-section {{ margin: 10px 0; padding: 10px 0; border-top: 1px solid #eee; }}
.topic-bar {{ height: 20px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }}
.topic-fill {{ padding-left: 8px; color: white; line-height: 20px; font-size: 0.9em; }}
.keyword {{ display: inline-block; background: #e3f2fd; padding: 4px 8px; margin: 2px; border-radius: 12px; }}
</style>
"""
        if self.map:
            self.map.get_root().html.add_child(folium.Element(html))

    def _add_custom_js(self):
        js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Custom JS for marker filtering or highlighting
});
</script>
"""
        self.map.get_root().html.add_child(folium.Element(js))

    def generate_map(self, businesses_df: pd.DataFrame, grid_analyzer: GridAnalyzer) -> folium.Map:
        # Set proper initial map location
        avg_lat = businesses_df['latitude'].mean()
        avg_lon = businesses_df['longitude'].mean()
        self.map = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, 
                            width=self.config.map_width, height=self.config.map_height)
        
        # Add cluster polygons and markers
        for cid, cluster in grid_analyzer.clusters.items():
            if len(cluster["business_ids"]) >= self.config.cluster_params["min_count"]:
                # Add convex hull polygon
                if cluster["hull_vertices"]:
                    folium.Polygon(
                        locations=[[p[0], p[1]] for p in cluster["hull_vertices"]],
                        color=self.config.get_topic_color(cluster["dominant_topic"]),
                        fill=True,
                        fill_opacity=0.2,
                        popup=self.popup_generator.create_cluster_popup(cid, cluster, grid_analyzer)
                    ).add_to(self.map)
                
                # Add cluster marker
                folium.Marker(
                    location=[cluster["center"]["lat"], cluster["center"]["lon"]],
                    icon=folium.Icon(color='gray', icon='cloud'),
                    popup=self.popup_generator.create_cluster_popup(cid, cluster, grid_analyzer)
                ).add_to(self.map)

        # Add individual business markers with clustering
        marker_cluster = MarkerCluster().add_to(self.map)
        for _, business in businesses_df.iterrows():
            popup_content = self.popup_generator.create_business_popup(
                business["business_id"], business.to_dict())
            folium.Marker(
                location=[business["latitude"], business["longitude"]],
                popup=folium.Popup(popup_content, max_width=400),
                icon=folium.Icon(color="blue", icon="utensils")
            ).add_to(marker_cluster)

        # Add map controls
        Fullscreen().add_to(self.map)
        LocateControl().add_to(self.map)
        MeasureControl().add_to(self.map)
        
        # Add layer control
        self._add_layer_controls(marker_cluster)
        return self.map

    def _add_layer_controls(self, marker_cluster):
        """Adds layer toggle controls to the map"""
        control_html = '''
        <div id="layer-control" style="position: fixed; top: 10px; right: 10px; z-index: 1000;
             background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
            <h4 style="margin:0 0 8px 0;">Toggle Layers</h4>
            <label style="display: block; margin: 3px 0;">
                <input type="checkbox" id="toggleBusinesses" checked 
                       onclick="toggleLayer('businesses')"> Individual Restaurants
            </label>
            <label style="display: block; margin: 3px 0;">
                <input type="checkbox" id="toggleClusters" checked 
                       onclick="toggleLayer('clusters')"> Clusters
            </label>
        </div>
        
        <script>
        function toggleLayer(layerType) {
            const map = document.getElementById('map');
            if(layerType === 'businesses') {
                Array.from(map.getElementsByClassName('marker-cluster')).forEach(e => 
                    e.style.display = document.getElementById('toggleBusinesses').checked ? '' : 'none');
            }
            if(layerType === 'clusters') {
                Array.from(map.getElementsByClassName('leaflet-marker-icon')).filter(e => 
                    e.src.includes('cloud')).forEach(marker => 
                    marker.style.display = document.getElementById('toggleClusters').checked ? '' : 'none');
                Array.from(map.getElementsByClassName('leaflet-interactive')).forEach(poly => 
                    poly.style.display = document.getElementById('toggleClusters').checked ? '' : 'none');
            }
        }
        </script>
        '''
        self.map.get_root().html.add_child(folium.Element(control_html))

    def save_map(self, out_path: str):
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        self.map.save(out_path)
        self.logger.info(f"Map saved to: {out_path}")

    def open_map(self, out_path: str):
        webbrowser.open("file://"+os.path.abspath(out_path))

# --- SimilarityCalculator ---
class SimilarityCalculator:
    def __init__(self, config: UDAConfig):
        self.config = config
        self.logger = logging.getLogger("UDA.SimilarityCalculator")
    
    def calculate_business_similarities(self, business_data: Dict[str,Dict]) -> Dict[str,List[Tuple[str,float]]]:
        vectors={}
        for bid,bd in business_data.items():
            vectors[bid]= self._feature_vector(bd)
        sims={}
        for bid,vec in vectors.items():
            simlist=[]
            for obid,ovec in vectors.items():
                if obid==bid:
                    continue
                simlist.append((obid,self._cosine_similarity(vec,ovec)))
            simlist.sort(key=lambda x:x[1], reverse=True)
            sims[bid]= simlist[:5]
        return sims

    def _feature_vector(self, data: Dict) -> np.ndarray:
        sentiment = data.get("sentiment",{})
        s = np.array([sentiment.get(k,0) for k in ["positive","neutral","negative"]])
        topics = data.get("topics",{})
        t= np.array([topics.get(x,0) for x in self.config.topics])
        v= np.concatenate([s,t])
        norm= np.linalg.norm(v)
        return v/norm if norm else v

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1,vec2))

# --- CityAnalytics ---
class CityAnalytics:
    def __init__(self, config: UDAConfig):
        self.config = config
        self.logger = logging.getLogger("UDA.CityAnalytics")
    
    def analyze_city_data(self, businesses_df: pd.DataFrame, reviews_df: pd.DataFrame) -> Dict[str,Any]:
        return {
            "most_discussed_topics": self._get_most_discussed_topics(businesses_df),
            "top_rated": self._get_top_restaurants(businesses_df, top=True),
            "low_rated": self._get_top_restaurants(businesses_df, top=False),
            "key_complaints": self._get_key_complaints(reviews_df)
        }

    def _get_most_discussed_topics(self, df: pd.DataFrame) -> List[Dict[str,Any]]:
        # Count how many times a topic appears in the 'topics' dict
        cnt=defaultdict(int)
        for _,row in df.iterrows():
            for t in row.get("topics",{}):
                cnt[t]+=1
        sorted_t= sorted(cnt.items(), key=lambda x:x[1], reverse=True)
        return [{"topic":k,"count":v} for k,v in sorted_t]

    def _get_top_restaurants(self, df: pd.DataFrame, top: bool=True, limit:int=10) -> List[Dict[str,Any]]:
        # only consider businesses with at least 5 reviews
        df_filt = df[df["review_count"]>=5]
        if df_filt.empty:
            return []
        df_srt = df_filt.sort_values(by="stars", ascending= not top)
        return df_srt.head(limit).to_dict("records")

    def _get_key_complaints(self, reviews_df: pd.DataFrame) -> List[Dict[str,Any]]:
        comp=Counter()
        for _, row in reviews_df.iterrows():
            if row.get("stars",3)<3:
                # parse text
                text= row.get("text","").lower()
                tokens= word_tokenize(text)
                for w in tokens:
                    if w in ["slow","rude","dirty","overpriced"]:
                        comp[w]+=1
        return [{"complaint":k,"count":v} for k,v in comp.most_common(5)]

# --- UrbanDishAnalyzer ---
class UrbanDishAnalyzer:
    def __init__(self, config: UDAConfig):
        self.config = config
        self.logger = setup_logging()
        self.cache = Cache(config)
        self.data_loader = DataLoader(config, self.cache)
        self.aspect_extractor = AspectExtractor(config, self.cache)
        self.grid_analyzer = GridAnalyzer(config, self.cache)
        self.similarity_calculator = SimilarityCalculator(config)
        self.city_analytics = CityAnalytics(config)
        self.map_visualizer = None
        self.businesses_df = None
        self.reviews_df = None

    def load_zero_shot_model(self):
        self.aspect_extractor.load_zero_shot_model()
        self.aspect_extractor.load_sentiment_model()

    def run(self):
        try:
            bus_df = self.data_loader.load_business_data()
            rev_df = self.data_loader.load_reviews(set(bus_df["business_id"].unique()))
            bus_df = bus_df[bus_df["business_id"].isin(rev_df["business_id"])].copy()  # Avoid SettingWithCopyWarning
            self.logger.info(f"Loaded {len(bus_df)} businesses after filtering reviews")

            self.load_zero_shot_model()
            bus_df = self.aspect_extractor.classify_reviews(rev_df, bus_df)

            self.grid_analyzer.load_business_data(bus_df, rev_df)
            self.grid_analyzer.calculate_clusters(bus_df, rev_df)
            self.logger.info("Grid clusters calculated")

            # Attempt hybrid clustering only if at least one cluster has >= min_count
            if any(len(cl["business_ids"]) >= self.config.cluster_params["min_count"] 
                   for cl in self.grid_analyzer.clusters.values()):
                self.grid_analyzer.hybrid_clusters(eps=0.05, min_samples=1)
                self.logger.info("Hybrid clustering complete")
            else:
                self.logger.warning("No valid grid clusters found; skipping hybrid clustering.")

            from types import SimpleNamespace  # to replicate your map visual classes
            # Initialize the map visualizer
            self.map_visualizer = MapVisualizer(self.config, self.grid_analyzer.business_data)
            m = self.map_visualizer.generate_map(bus_df, self.grid_analyzer)
            if m:
                out_path = os.path.join("output", f"{self.config.city}_restaurant_clusters.html")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                self.map_visualizer.save_map(out_path)
                self.map_visualizer.open_map(out_path)
                self.logger.info("Map generation complete")
            else:
                self.logger.warning("Map generation failed")

            # Similarities
            sims = self.similarity_calculator.calculate_business_similarities(self.grid_analyzer.business_data)
            self.grid_analyzer.business_similarities = sims

            # Analytics
            analytics = self.city_analytics.analyze_city_data(bus_df, rev_df)
            self.map_visualizer.add_analytics_panel(analytics)

        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            self.logger.error(traceback.format_exc())

# --- Main Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanDishAnalyzer")
    parser.add_argument("--city", type=str, default="tucson", help="City to analyze")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--sample_limit", type=int, default=None, help="Sample limit for data loading")
    parser.add_argument("--max_reviews", type=int, default=50, help="Maximum number of reviews to load")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--no_cache", dest="use_cache", action="store_false", help="Do not use cached data")
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    config = UDAConfig(
        city=args.city,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        sample_limit=args.sample_limit,
        max_reviews=args.max_reviews
    )
    download_nltk_resources()
    analyzer = UrbanDishAnalyzer(config)
    analyzer.run()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()