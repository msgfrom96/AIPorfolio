# Urban Dish Analyzer (UDA)

A powerful tool for analyzing restaurant reviews in cities to discover food trends, sentiment patterns, and restaurant clusters.

## Business Problem Definition

Urban Dish Analyzer addresses a critical business challenge in the restaurant industry: **How can restaurant owners, food service companies, and city planners gain data-driven insights about local food trends and customer preferences?**

This problem is significant because:
- Restaurant owners need to understand local competition and customer sentiment to improve their offerings
- Food service companies require market analysis for expansion decisions
- City planners benefit from understanding restaurant density and cuisine distribution
- Customers want personalized recommendations based on their preferences

By analyzing large-scale Yelp review data through natural language processing, UDA transforms unstructured customer feedback into actionable business intelligence.

## Overview

Urban Dish Analyzer extracts insights from Yelp restaurant reviews using natural language processing and machine learning. It identifies topics of discussion, analyzes sentiment, and clusters restaurants based on location and review content, then visualizes these insights on an interactive map.

## Features

- **Review Analysis**: Extracts key topics, sentiment, and keywords from restaurant reviews
- **Hybrid Clustering**: Groups restaurants using both geographic proximity and review content similarity
- **Interactive Visualization**: Generates an interactive map with:
  - Restaurant clusters with convex hull boundaries
  - Color-coded markers based on dominant topics
  - Detailed popup information for both individual restaurants and clusters
  - Sentiment analysis results
  - Top keywords from reviews
  - Similar restaurant recommendations
- **City Analytics**: Provides city-level insights such as:
  - Most discussed topics across all restaurants
  - Top-rated and low-rated establishments
  - Common complaints and issues

## Connecting Business Problems to Technical Approach

| Business Need | Technical Solution | Implementation Detail |
|---------------|-------------------|------------------------|
| **Restaurant Differentiation** | Zero-shot topic classification | The `AspectExtractor` class uses `valhalla/distilbart-mnli-12-3` to categorize reviews into key business aspects like food quality, service, and ambiance |
| **Customer Sentiment Understanding** | Fine-tuned sentiment analysis | `sentiment_pipeline` using RoBERTa with star-rating calibration generates granular sentiment metrics for specific aspects of business |
| **Geographic Market Analysis** | Hybrid spatial clustering | `GridAnalyzer` implements grid-based clustering with convex hull visualization to identify restaurant hotspots and gaps |
| **Competitive Analysis** | TF-IDF + cosine similarity | The `SimilarityCalculator` class computes business similarity with content-based nearest neighbors to identify competition |
| **Trend Identification** | Keyword extraction with POS tagging | spaCy-based extraction in the `_filter_keywords` method isolates meaningful terms to track emerging food trends |
| **Decision Support for Expansion** | City-level analytics engine | `CityAnalytics` class aggregates insights across businesses for macro-level strategy |

## Exploratory Data Analysis

The UDA performs comprehensive exploratory data analysis on Yelp business and review data:

1. **Data Cleaning and Preprocessing**:
   - Removes duplicates and handles missing values
   - Filters for relevant business categories
   - Normalizes text data (lowercase, punctuation removal)
   - Structures hierarchical business-to-review relationships

2. **Statistical Analysis**:
   - Calculates distribution of ratings, reviews per business
   - Identifies geographic density patterns of restaurants
   - Analyzes review length and content complexity

3. **Visualization**:
   - Generates heatmaps of restaurant density
   - Plots sentiment distribution across neighborhoods
   - Creates word frequency visualizations for common topics

Our EDA process reveals key patterns in restaurant data that inform the subsequent NLP analysis.

## NLP Methodology

The UDA employs several advanced NLP techniques, each chosen to address specific business challenges:

1. **Zero-Shot Text Classification** (using `valhalla/distilbart-mnli-12-3`):
   - **Business Application**: Enables restaurant owners to understand which aspects of their business customers discuss most
   - **Implementation**: Multi-label classification of review text into predefined topics
   - **Topics include**: food quality, service quality, ambiance, price, etc.
   - **Technical Enhancement**: Temperature scaling for calibrated confidence scores ensures reliable insights

2. **Sentiment Analysis** (using `cardiffnlp/twitter-roberta-base-sentiment`):
   - **Business Application**: Helps identify specific strengths and weaknesses across different aspects of service
   - **Implementation**: Fine-tuned RoBERTa model to classify sentiment as positive, negative, or neutral
   - **Technical Enhancement**: Integrated with star rating data for more nuanced analysis
   - **Business Value**: Enables targeted improvement strategies focused on problematic areas

3. **Keyword Extraction** (using spaCy):
   - **Business Application**: Reveals specific menu items, staff behaviors, or amenities that drive customer satisfaction
   - **Implementation**: Part-of-speech tagging to identify relevant nouns, adjectives, and verbs
   - **Technical Enhancement**: TF-IDF vectorization to identify distinguishing terms
   - **Business Value**: Pinpoints exact factors affecting customer experience

4. **Text Similarity Analysis**:
   - **Business Application**: Enables competitive analysis and restaurant recommendations
   - **Implementation**: Cosine similarity on combined feature vectors
   - **Business Value**: Helps restaurants understand their market position and closest competitors

Each NLP technique was selected based on its ability to extract specific business-relevant insights from unstructured review text.

## Business Insights & Results

The UDA transforms technical NLP outputs into actionable business insights:

1. **Competitive Landscape Analysis**:
   - **Technical Source**: Hybrid clustering algorithm in `GridAnalyzer`
   - **Business Output**: Identifies restaurant clusters and their dominant topics
   - **Actionable Insight**: Restaurant owners can identify under-served areas or over-saturated markets
   - **Implementation Path**: Grid-based spatial clustering + content-based merging → convex hull visualization

2. **Sentiment Drivers**:
   - **Technical Source**: Sentiment analysis + keyword correlation in `AspectExtractor`
   - **Business Output**: Correlates specific keywords with positive/negative sentiment
   - **Actionable Insight**: Identifies exactly what customers love or hate about similar establishments
   - **Implementation Path**: RoBERTa sentiment classification → keyword correlation → business-level aggregation

3. **Trend Identification**:
   - **Technical Source**: Temporal analysis of extracted keywords in `CityAnalytics`
   - **Business Output**: Maps emerging food preferences and cuisine trends
   - **Actionable Insight**: Helps restaurants adapt menus to evolving customer preferences
   - **Implementation Path**: spaCy POS tagging → TF-IDF analysis → topic clustering → trend visualization

4. **Decision Support**:
   - **Technical Source**: Integrated outputs from all analysis components
   - **Business Output**: Comprehensive dashboard with actionable recommendations
   - **Actionable Insight**: Specific, prioritized improvement areas for business owners
   - **Implementation Path**: Multi-modal data fusion → interactive visualization → analytics panel

Each business insight is directly derived from a specific technical implementation, ensuring that all technical work directly addresses business needs.

## Installation

### Prerequisites

- Python 3.8+
- Yelp dataset files (`business.json` and `review.json`)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/msgfrom96/urban-dish-analyzer.git
   cd urban-dish-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download required NLTK resources (will be done automatically on first run, but can be done manually):
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

```bash
python UDA/Final\ UDA.py --city tucson --data_dir path/to/yelp_data
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--city` | City to analyze | tucson |
| `--cache_dir` | Directory for caching data | .cache |
| `--data_dir` | Directory containing Yelp data files | data |
| `--sample_limit` | Maximum number of businesses to load | None (all) |
| `--max_reviews` | Maximum number of reviews to load | 50 |
| `--use_cache` | Use cached data if available | True |
| `--no_cache` | Do not use cached data | False |

### Example

Analyze restaurants in Phoenix with up to 100 reviews per restaurant:

```bash
python UDA/Final\ UDA.py --city phoenix --max_reviews 100 --data_dir data
```

## Technical Design Decisions

Each technical design choice in UDA is driven by specific business requirements:

1. **Grid-Based Clustering with Hybrid Refinement**
   - **Business Need**: Understand restaurant distribution with both geographic and thematic grouping
   - **Why This Approach**: Traditional clustering alone doesn't account for both physical proximity and restaurant similarity
   - **Business Benefit**: Reveals both geographic hotspots and thematically similar restaurant groups

2. **Zero-Shot Classification over Fine-Tuning**
   - **Business Need**: Analyze diverse topics without requiring labeled training data
   - **Why This Approach**: Allows flexibility in topics without costly annotation
   - **Business Benefit**: Can quickly adapt to analyzing new aspects of restaurant experience

3. **Interactive Web-Based Visualization**
   - **Business Need**: Make insights accessible to non-technical stakeholders
   - **Why This Approach**: Folium's interactive maps offer intuitive exploration without specialized tools
   - **Business Benefit**: Increases adoption of insights across all levels of an organization

4. **Caching System**
   - **Business Need**: Enable repeated analysis without redundant processing
   - **Why This Approach**: Minimizes computational costs while allowing incremental updates
   - **Business Benefit**: Reduces operational costs and enables continuous monitoring

## Data Requirements

The analyzer expects Yelp dataset files in the specified `data_dir`:
- `business.json` or `yelp_academic_dataset_business.json`: Contains business information
- `review.json` or `yelp_academic_dataset_review.json`: Contains review information

You can download the Yelp dataset from [Yelp Dataset](https://www.yelp.com/dataset).

## Configuration

Advanced configurations can be modified in the `UDAConfig` class within the script:

- **Topics**: Change the topics to analyze in reviews
- **Clustering parameters**: Adjust the sensitivity of restaurant clustering
- **Sentiment thresholds**: Modify how sentiment is categorized
- **Map appearance**: Change visualization colors and styles

## Output

The analyzer generates an interactive HTML map in the `output` directory named `[city]_restaurant_clusters.html`. This map will automatically open in your default web browser when the analysis is complete.

## Logging

Logs are stored in the `logs` directory. Check `logs/uda.log` for detailed information about the analysis process, errors, and warnings.

## Code Structure & Quality

The Urban Dish Analyzer follows software engineering best practices:

### Modular Architecture
The code is organized into specialized classes with clear responsibilities:
- `DataLoader`: Handles data retrieval and caching
- `AspectExtractor`: Implements NLP functionality
- `GridAnalyzer`: Manages spatial clustering
- `MapVisualizer`: Creates interactive visualizations
- `CityAnalytics`: Computes city-level insights

### Design Patterns
- **Singleton Pattern**: For configuration management
- **Factory Method**: For creating different analysis components
- **Observer Pattern**: For updating visualization based on data changes

### Error Handling
- Comprehensive try/except blocks with detailed error messages
- Graceful degradation with fallback options
- Detailed logging at multiple severity levels

### Performance Optimization
- Efficient batch processing of review data
- Caching system for intermediate results
- Memory management for large datasets (chunking)
- GPU acceleration where available

### Documentation
- Docstrings for all classes and methods
- Type annotations throughout the codebase
- Inline comments explaining complex algorithms

## Performance Considerations

- Analysis of large cities may require significant memory and CPU resources
- Using a GPU can significantly speed up the sentiment and topic analysis phases
- The `sample_limit` and `max_reviews` parameters can be used to limit resource usage
- Consider using the cache for repeated analyses of the same data
