# 📚 Basic libraries
import pandas as pd
import ast

def main():
    """
    Preprocess the reviews and create a new dataframe with combined features.
    """
    df = load_dataset()
    df['authors'] = df['authors'].apply(clean_author)
    df['shortened_text'] = df['all_texts_processed'].apply(lambda x: limit_text_length(x, max_words=300))
    df['genre_label'] = df.apply(reverse_one_hot_encoding, axis=1)
    df["combined_features"] = df.apply(combine_features, axis=1)
    df['combined_features'] = df['combined_features'].str.lower()

    # Creates a new dataframe with the selected columns
    new_df = df[['title', 'authors', 'avg_score', 'sentiment_score', 'cluster_label', 'genre_label', 'combined_features']]

    # Save the new version with combiined features
    new_df.to_csv("books_combined_features.csv", index=False)
    print("✅ New dataframe saved!")

def load_dataset():
    data = pd.read_csv("../datasets/cluster_dataset.csv")
    return data.copy()

def clean_author(author_str):
    """
    Cleans the author string to a more readable format.
    """
    try:
        # Convert string to list
        authors = ast.literal_eval(author_str)
        if isinstance(authors, list):
            return ", ".join(authors)
    except:
        pass
    return ""

def limit_text_length(text, max_words=300):
    """
    Limits the text to the first 'max_words' words.
    """
    if isinstance(text, str):  # Check if the text is a string
        words = text.split()  # Split by whitespace
        return " ".join(words[:max_words])
    else:
        return ""

def reverse_one_hot_encoding(row):
    """
    Reverse the one-hot encoding to get the genre.
    """
    genres = []
    for column in row.index:
        if column.startswith("genre_") and row[column] == 1:  # Adjust if your column names are different
            genre = column.replace("genre_", "")  # Remove the 'genre_' prefix
            genres.append(genre)
    return ", ".join(genres)

def combine_features(row):
    """
    Combine the title, author, genre, and reviews into a single string.
    """
    title = row['title'] if pd.notnull(row['title']) else ""
    author = row['authors'] if pd.notnull(row['authors']) else ""
    genre = row['genre_label'] if 'genre_label' in row and pd.notnull(row['genre_label']) else ""

    reviews = row['shortened_text'] if 'shortened_text' in row and pd.notnull(row['shortened_text']) else ""

    return f"{title}. Written by {author}. Genre: {genre}. Reviews: {reviews}"

if __name__ == "__main__":
    main()