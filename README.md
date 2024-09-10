# Cryptocurrency Sentiment and Emotion Analysis

## Project Motivation

This project aims to conduct a comprehensive sentiment and emotion analysis on tweets related to cryptocurrency, specifically focusing on Bitcoin. By leveraging machine learning models, sentiment classification techniques, and pre-trained emotion detection models, we extract valuable insights into public opinion, sentiment trends, and emotional responses within cryptocurrency discussions.

## Libraries Used

- **Pandas**: For data manipulation.
- **Numpy**: For numerical operations.
- **NLTK (Natural Language Toolkit)**: For text preprocessing, stemming, and sentiment analysis.
- **Scikit-learn**: For machine learning models, TF-IDF vectorization, and evaluation metrics.
- **Matplotlib**: For data visualization.
- **Transformers**: For emotion detection using pre-trained models.
- **Tweepy**: For fetching tweets from the Twitter API.

## Data Overview

The dataset consists of 197,753 tweets related to Bitcoin and other cryptocurrencies. Due to computational constraints, we subsetted the data to 500 randomly sampled tweets for this analysis.

### Sample Data

The dataset contains the following columns:
- `text`: The text of the tweet.
- `sentiment_score`: Compound score derived from sentiment analysis.
- `sentiment`: Binary sentiment classification (1 for positive, 0 for negative).
- `emotions`: Emotion labels extracted from the emotion detection model.

## Text Preprocessing

Text preprocessing was performed using the following steps:

1. **Stemming**: Using the **Porter Stemmer** from NLTK to reduce words to their base form (e.g., "running" becomes "run").
2. **Stopword Removal**: Common English stopwords (e.g., "the", "and") were removed to reduce noise.
3. **Tokenization**: Text was split into individual words for further processing.

### Example of Stemming Function

```python
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
