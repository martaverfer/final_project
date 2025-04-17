# Import libraries
import streamlit as st 
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 Book Recommender", layout="centered")

# --- Load API key ---
api_key = st.secrets["OPENAI_API_KEY"] 

# ===================================
#              FUNCTIONS
# ===================================

@st.cache_data
def load_data():
    """Load the clustered dataset"""
    return pd.read_csv("../datasets/books_combined_features.csv")

df = load_data()

@st.cache_resource
def get_tfidf_model_and_matrix(texts):
    """Create TF-IDF vectorizer and matrix"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_tfidf_model_and_matrix(df['combined_features'])

@st.cache_data(show_spinner=False)
def generate_summary_with_openai_from_title(title, author=None):
    """
    Generate summary using OpenAI's GPT-3.5 or GPT-4 model using the new API format.
    """
    prompt = f"Give me a 2-sentence summary of the book titled '{title}'"
    if pd.notnull(author) and author != "[]":
        prompt += f" by {author}"
    prompt += "."

    try:
        # Use the new API method for completions
        client = openai.OpenAI()
        response =  client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                 {"role": "system", "content": "You are a helpful literary assistant."},
                 {"role": "user", "content": prompt}
             ],
            max_tokens=100,  # Limit the tokens to ensure a short summary
            temperature=0.7  # Control the response's creativity
        )
        
        # Extract summary from response
        summary = response.choices[0].message.content.strip()
        
        if not summary:
            return "❌ Could not generate summary."
        return summary
    
    except Exception as e:
        # Handle any exception and show an error message
        st.error(f"Error generating summary: {str(e)}")
        return "❌ Could not generate summary."


# ===================================
#              STREAMLIT
# ===================================

st.title("📚 Smart Book Recommender")
st.write("Tell me what kind of book you're looking for and get a personalized recommendation & summary!")

# User input
user_input = st.text_input("What kind of book do you want?", placeholder="e.g. Recommend me something romantic and fun")

# Cluster filter
use_cluster = st.checkbox("🎯 Filter by specific cluster")
if use_cluster:
    available_clusters = sorted(df['cluster_label'].unique())
    cluster_select = st.selectbox("Select a cluster", available_clusters)
    df_filtered = df[df['cluster_label'] == cluster_select].copy()
    tfidf_matrix_filtered = tfidf_matrix[df_filtered.index]
else:
    df_filtered = df.copy()
    tfidf_matrix_filtered = tfidf_matrix

# Matching logic
if user_input:
    user_vector = vectorizer.transform([user_input.lower()])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix_filtered).flatten()
    df_filtered['similarity'] = similarity_scores

    top_books = df_filtered[df_filtered['similarity'] > 0.2].sort_values("similarity", ascending=False).head(2)

    st.markdown("### 🎯 Top Book Recommendations:")

    for i, (_, row) in enumerate(top_books.iterrows(), 1):
        st.subheader(f"📘 Recommendation #{i}: {row['title']}")
        st.write(f"👤 Author(s): {row['authors']}")
        st.write(f"⭐ Rating: {row['avg_score']} | 😊 Sentiment: {round(row['sentiment_score'], 2)}")

        with st.spinner("Generating summary..."):
            ai_summary = generate_summary_with_openai_from_title(row['title'], row['authors'])

        st.write(f"📝 **Summary**: {ai_summary}")
        st.write(f"🏷️ **Cluster**: {row['cluster_label']}")
        st.markdown("---")