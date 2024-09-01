import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data
data = pd.read_csv("books_data.csv")

# Title for the Streamlit app
st.title("Book Recommendation System")



# Display top 10 authors by the number of books
st.subheader("Top 10 Authors by Number of Books")


# Display top 10 authors by the number of books
st.subheader("Top 10 Authors by Number of Books")
top_authors = data['authors'].value_counts().head(10)
st.write(top_authors)

# Convert 'average_rating' to a numeric data type
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')

# Create a new column 'book_content' by combining 'title' and 'authors'
data['book_content'] = data['title'] + ' ' + data['authors']

# Transform the text-based features into numerical vectors using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])

# Compute the cosine similarity between books
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend books
def recommend_books(book_title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = data[data['title'] == book_title].index[0]

    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended books
    return data['title'].iloc[book_indices]

# User selection for the book title
st.subheader("Book Recommendation")
book_title = st.selectbox("Select a book title", data['title'].unique())

if book_title:
    # Display recommended books
    recommended_books = recommend_books(book_title)
    st.write("Top 10 recommended books:")
    st.write(recommended_books)
