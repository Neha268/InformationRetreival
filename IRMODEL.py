#!/usr/bin/env python
# coding: utf-8


# ##### IR system based on the vector space model

# **Recommendation Systems using Vector space model**:
# - Recommendation systems to suggest similar items (e.g., movies). By identifying the most similar movies based on cosine similarity, we can recommend movies with similar genres, or keywords.
# - The given code is used to recommend top-5 movies based on title using Vector space model.
# - Here I have used the **TfidfVectorizer and cosine_similarity** 

# so firstly we need to import necessary libraries 

# - pandas : This is used for data manipulation and analysis. It is commonly used for reading, writing, and processing datasets in tabular formats.
# - sklearn.feature_extraction.text.TfidfVectorizer: This module provides a way to convert a collection of text documents into a matrix of TF-IDF features. TF-IDF (Term Frequency-Inverse Document Frequency) is a common technique for measuring the importance of words in a document or set of documents.
# - from sklearn.metrics.pairwise import cosine_similarity: This import statement brings the cosine_similarity function into Python environment, allowing to use it to measure similarity between pairs of vectors.

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
keywords = pd.read_csv('keywords.csv')
metadata = pd.read_csv('movies_metadata.csv')



#  **Convert ID to String**:
# - metadata['id'] = metadata['id'].astype(str): This step converts the 'id' column in the metadata DataFrame to a string type, ensuring consistency. This is useful when we intend to merge datasets on a common field where data types must match.
# - keywords['id'] = keywords['id'].astype(str): Similarly, this converts the 'id' column in the keywords DataFrame to a string.
# **Merging DataFrames**:
# - movies = metadata.merge(keywords, on='id'): This merges the metadata and keywords DataFrames based on the 'id' column. After merging, the resulting movies DataFrame contains data from both DataFrames for matching IDs.
# **Filling Missing Values**:
# - movies['overview'] = movies['overview'].fillna(''): Fills missing values in the 'overview' column with empty strings. This approach is often used when dealing with text data to avoid issues with operations that require non-null values, such as TF-IDF vectorization.
# - movies['keywords'] = movies['keywords'].fillna(''): This fills missing values in the 'keywords' column with empty strings.

# In[2]:


# Perform necessary data preprocessing
metadata['id'] = metadata['id'].astype(str)  # Convert id to string for consistency
keywords['id'] = keywords['id'].astype(str)
movies = metadata.merge(keywords, on='id')

# Fill missing values with empty strings for TF-IDF vectorizer
movies['overview'] = movies['overview'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')



# **TfidfVectorizer with Stop Words**:
# - tfidf_vectorizer = TfidfVectorizer(stop_words='english'): This initializes a TfidfVectorizer with English stop words removed. Stop words are common words (like "the", "and", "is") that typically do not add significant meaning and can be excluded to reduce noise in text analysis.
# **Creating the TF-IDF Matrix**:
# - tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'] + ' ' + movies['keywords']): This line concatenates the 'overview' and 'keywords' columns into a single text string for each movie and applies TF-IDF vectorization.
# - The fit_transform method computes the TF-IDF values for each term and converts the text data into a sparse matrix, where each row represents a movie and each column represents a term with its TF-IDF score.

# **Cosine Similarity Calculation**:
# - cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix): This computes the cosine similarity between all pairs of movies based on the TF-IDF matrix. The result is a square matrix where the value at row i, column j indicates the similarity between movie i and movie j.

# In[3]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'] + ' ' + movies['keywords'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)




# - **Case-Insensitive Title Matching**: Converts the input title and the 'title' column to lowercase for case-insensitive comparison.
# - **Title Existence Check**: Checks if the provided title exists in the movies dataset. If it doesn't, returns an appropriate message.
# - **Index Out-of-Bounds Handling**: Uses a try-except block to catch IndexError in case the index retrieval fails.
# Returning Results: If everything goes smoothly, the function returns a list of recommended movie titles. Otherwise, it provides a meaningful error message.

# In[22]:


def get_recommendations(title, cosine_sim, movies, top_n=5):

    title = title.lower()


    if not any(movies['title'].str.lower() == title):

        return f"Movie title '{title}' not found in the dataset."

    # Get the index of the movie that matches the title
    try:
        idx = movies[movies['title'].str.lower() == title].index[0]
    except IndexError:
        return f"Error: Unable to find index for the given title '{title}'."

    # Pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # top N most similar movies 
    sim_scores = sim_scores[1:top_n + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top N most similar movies
    return movies['title'].iloc[movie_indices].tolist()

title = 'Superman'
recommendations = get_recommendations(title, cosine_sim, movies, top_n=5)

if isinstance(recommendations, list):
    print("Recommendations:")
    for movie in recommendations:
        print(movie)  
else:
    print(recommendations)  


# The get_recommendations function appears to be a useful tool for providing movie recommendations based on cosine similarity. 

# 
