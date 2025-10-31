# Movie Recommender System

A content-based movie recommendation system built with Streamlit and scikit-learn. This application suggests movies based on similar content, tags, and descriptions.

## Features

- 🎬 Interactive Streamlit web interface
- 🎯 Content-based movie recommendations
- 📊 Cosine similarity algorithm for finding similar movies
- 🔍 Easy-to-use movie selector
- 💡 Top 10 movie recommendations

## Installation

1. Clone or download this repository

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Select a movie from the dropdown menu

4. Click "Get Recommendations" to see similar movies

## How It Works

The recommendation system uses:

- **Content-based filtering**: Analyzes movie tags and descriptions
- **Count Vectorization**: Converts text data into numerical vectors (using 5000 most frequent features)
- **Cosine Similarity**: Measures similarity between movies based on their content vectors
- **Caching**: Similarity matrix is cached for faster subsequent runs

## Project Structure

```
piyush_ml_project/
├── app.py                 # Main Streamlit application
├── recommender.py         # Recommendation engine with similarity matrix
├── movie_dict.pkl         # Pickle file containing movie data
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Dependencies

- Streamlit: Web application framework
- Pandas: Data manipulation
- scikit-learn: Machine learning algorithms (CountVectorizer, cosine_similarity)
- NumPy: Numerical computing

## Notes

- The system precomputes and caches the similarity matrix for better performance
- First run will take longer as it computes the similarity matrix (4806 x 4806)
- Subsequent runs will be faster as the matrix is cached in `similarity_matrix.npy`
- Recommendations are based on movie tags and content similarity
- The more similar the content, the higher the recommendation score

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Delete `similarity_matrix.npy` if you want to regenerate the similarity matrix
3. Ensure `movie_dict.pkl` is in the project root directory

## License

This project is open source and available for educational purposes.
