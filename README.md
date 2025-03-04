# Urban Dish Analyzer (UDA)

A powerful tool for analyzing restaurant reviews in cities to discover food trends, sentiment patterns, and restaurant clusters.

## Overview

Urban Dish Analyzer extracts insights from Yelp restaurant reviews using natural language processing and machine learning. It identifies topics of discussion, analyzes sentiment, and clusters restaurants based on location and review content, then visualizes these insights on an interactive map.

![UDA Map Example](docs/map_example.png)

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

## Installation

### Prerequisites

- Python 3.8+
- Yelp dataset files (`business.json` and `review.json`)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/urban-dish-analyzer.git
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

## Performance Considerations

- Analysis of large cities may require significant memory and CPU resources
- Using a GPU can significantly speed up the sentiment and topic analysis phases
- The `sample_limit` and `max_reviews` parameters can be used to limit resource usage
- Consider using the cache for repeated analyses of the same data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Your Name
