import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_movies():
    """Load movies data from pickle file"""
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict())
    return movies


def create_similarity_matrix():
    """Create similarity matrix based on movie tags with caching"""
    movies = load_movies()
    
    # Check if cached similarity matrix exists
    similarity_cache_file = 'similarity_matrix.npy'
    
    if os.path.exists(similarity_cache_file):
        print("Loading cached similarity matrix...")
        similarity = np.load(similarity_cache_file)
    else:
        print("Creating similarity matrix (this may take a moment)...")
        # Create a CountVectorizer instance
        cv = CountVectorizer(max_features=5000, stop_words='english')
        
        # Fit and transform the tags
        vectors = cv.fit_transform(movies['tags']).toarray()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors)
        
        # Cache the similarity matrix for faster future loads
        np.save(similarity_cache_file, similarity)
        print("Similarity matrix created and cached!")
    
    return movies, similarity


def recommend_movies(movie_name, movies, similarity, top_n=5):
    """
    Recommend movies based on a selected movie
    
    Args:
        movie_name: Name of the movie to get recommendations for
        movies: DataFrame containing movie data
        similarity: Similarity matrix
        top_n: Number of recommendations to return
    
    Returns:
        List of recommended movie titles, or empty list if movie not found
    """
    try:
        # Find the index of the selected movie
        movie_indices = movies[movies['title'] == movie_name].index
        
        if len(movie_indices) == 0:
            return []
        
        movie_index = movie_indices[0]
        
        # Get similarity scores for this movie
        distances = similarity[movie_index]
        
        # Sort movies by similarity (excluding the movie itself)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        
        # Extract movie titles
        recommended_movies = []
        for i in movie_list:
            recommended_movies.append(movies.iloc[i[0]]['title'])
        
        return recommended_movies
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []


# Precompute similarity matrix (optional - for performance)
movies_df, similarity_matrix = create_similarity_matrix()

