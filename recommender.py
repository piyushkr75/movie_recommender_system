import pickle
import pandas as pd
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_movie_dict():
    """Create movie dictionary from original dataset files"""
    try:
        # Define the path to the zip file
        zip_path = r"c:\Users\Ayushsingh\Desktop\PIYUSHHHHHHH\movie_recommend\archive.zip"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Extract files if they don't exist
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip
            files = zip_ref.namelist()
            
            # Extract only if files don't exist
            for file in files:
                if not os.path.exists(os.path.join("data", file)):
                    zip_ref.extract(file, "data")
        
        # Read the movies dataset (assuming it's a CSV file)
        # Try different possible filenames
        possible_files = ['movies.csv', 'tmdb_5000_movies.csv', 'movie_metadata.csv']
        movies_file = None
        
        for filename in possible_files:
            if os.path.exists(os.path.join("data", filename)):
                movies_file = os.path.join("data", filename)
                break
        
        if movies_file is None:
            raise FileNotFoundError("Could not find movies dataset in the archive")
            
        # Read the movies data
        movies_df = pd.read_csv(movies_file)
        
        # Add movie_id column (using id column from TMDB dataset)
        movies_df['movie_id'] = movies_df['id']
        
        # Select and rename relevant columns
        movies_df = movies_df[[
            'movie_id',
            'title',
            'overview',
            'genres',
            'keywords',
            'vote_average',
            'vote_count',
            'release_date'
        ]]
        
        # Create the movies dictionary
        movies_dict = movies_df.to_dict('records')
        
        # Save the dictionary to pickle file
        with open('movie_dict.pkl', 'wb') as f:
            pickle.dump(movies_dict, f)
            
        return movies_dict
        
    except Exception as e:
        print(f"Error creating movie dictionary: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the archive file exists and is not corrupted")
        print("2. Check if you have write permissions in the current directory")
        print(f"3. Make sure pandas ({pd.__version__}) can read the data file")
        raise


def load_movies():
    """Load movies data from pickle file with error handling"""
    try:
        # If pickle file doesn't exist, create it from the original dataset
        if not os.path.exists('movie_dict.pkl'):
            print("Movie dictionary pickle file not found. Creating from original dataset...")
            movies_dict = create_movie_dict()
        else:
            # Try to load existing pickle file
            with open('movie_dict.pkl', 'rb') as file:
                try:
                    movies_dict = pickle.load(file)
                except (pickle.UnpicklingError, AttributeError) as e:
                    print("Error loading existing pickle file. Recreating from original dataset...")
                    movies_dict = create_movie_dict()
        
        try:
            movies = pd.DataFrame(movies_dict)
            # Ensure movie_id column exists (create from id if needed)
            if 'movie_id' not in movies.columns and 'id' in movies.columns:
                movies['movie_id'] = movies['id']
            return movies
        except Exception as e:
            raise RuntimeError(
                "Error converting movie data to DataFrame. The data might be in an unexpected format. "
                f"Original error: {str(e)}"
            ) from e
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the original dataset archive is accessible")
        print("2. Check if you have write permissions in the current directory")
        print("3. Verify the dataset format in the archive")
        print(f"Current pandas version: {pd.__version__}")
        raise


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

