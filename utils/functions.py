# 📚 Basic libraries
import pandas as pd

# 📝 Text Processing
import spacy
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Apply Preprocessing to 'all_texts'
spacy.require_gpu()

# Load the transformer-powered model
nlp = spacy.load("en_core_web_trf")

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define Genre Keyword Map
genre_keywords = {
    'Fiction': ['novel', 'story', 'thriller', 'romance', 'mystery', 'narrative', 'fantasy', 'fiction'],
    'Non-Fiction': ['biography', 'memoir', 'self-help', 'life', 'true', 'religion', 'cooking', 'sports', 'travel', 'body', 'fitness', 'relationships', 'music', 'criticism', 'philosophy'],
    'Academic': ['research', 'study', 'academic', 'analysis', 'paper', 'theory', 'history', 'economics', 'computers', 'science', 'education', 'art', 'medical', 'psychology'],
    "Children's/Young Adult": ['children', 'young', 'teen', 'kid', 'juvenile'],
    'Poetry/Drama': ['poem', 'poetry', 'drama', 'play', 'verse']
}

def split_text(text, max_length=1000000):
    """
    Split a long text into smaller chunks of a specified maximum length.
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def preprocess_text_spacy_trf(text_series):
    """
    Preprocess text using spaCy Transformers: uses contextualized tokenization and lemmatization.
    Removes stop words and non-alphabetic tokens.
    Handles large texts by splitting them into chunks if needed.
    """
    processed = []

    # Use tqdm for progress bar and display the estimated time remaining
    for text in tqdm(text_series, desc="Processing Texts", unit="text", ncols=100, mininterval=0.5):
        # Split the text if it's too long
        chunks = split_text(text)
        
        # Process each chunk in the text series (batch size of 64)
        chunk_processed = []
        for chunk in chunks:
            doc = next(nlp.pipe([chunk], batch_size=64))  # process the chunk in a batch
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_stop
            ]
            chunk_processed.append(" ".join(tokens))
        
        # Join the processed chunks back together
        processed_text = " ".join(chunk_processed)
        processed.append(processed_text)

    return pd.Series(processed, index=text_series.index)

def get_sentiment_vader(text):
    """
    Get sentiment score using VADER.
    """
    if pd.isnull(text): return 0  # Handle missing values
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score ranges from -1 to 1

def extract_keywords(text):
    """
    Split text into tokens and return a set of unique tokens.
    """
    return set(text.split())

def assign_genre_from_keywords(text):
    """
    Assign a genre to the text based on keyword matching.
    """
    text_keywords = extract_keywords(text)
    genre_scores = {}

    for genre, keywords in genre_keywords.items():
        match_count = len(set(keywords) & text_keywords)
        genre_scores[genre] = match_count

    if genre_scores:
        best_genre = max(genre_scores, key=genre_scores.get)
        if genre_scores[best_genre] > 0:
            return best_genre
    return 'Unknown'

def assign_genre_from_categories(cat_text):
    """
    Assign a genre based on the categories column.
    """
    if pd.isnull(cat_text):
        return 'Unknown'

    if isinstance(cat_text, list):
        cat_text = " ".join(cat_text)
    
    text_keywords = set(cat_text.lower().split())
    genre_scores = {}

    for genre, keywords in genre_keywords.items():
        match_count = len(set(keywords) & text_keywords)
        genre_scores[genre] = match_count

    if genre_scores:
        best_genre = max(genre_scores, key=genre_scores.get)
        if genre_scores[best_genre] > 0:
            return best_genre
    return 'Unknown'


