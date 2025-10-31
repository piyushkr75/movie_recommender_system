import streamlit as st
from recommender import movies_df, similarity_matrix, recommend_movies

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title('🎬 Movie Recommender System')
st.markdown("---")
st.markdown("Select a movie and get personalized recommendations based on similar content!")

# Load movies
movies = movies_df

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📽️ Select a Movie")
    selected_movie_name = st.selectbox(
        "Choose a movie you like:",
        movies['title'].values,
        key="movie_selector"
    )
    
    # Show selected movie info
    if selected_movie_name:
        selected_movie = movies[movies['title'] == selected_movie_name].iloc[0]
        st.markdown("### Selected Movie:")
        st.info(f"**{selected_movie_name}** (ID: {selected_movie['movie_id']})")

with col2:
    st.subheader("🎯 Recommendations")
    
    if st.button("🔍 Get Recommendations", type="primary"):
        if selected_movie_name:
            with st.spinner('Finding similar movies...'):
                recommendations = recommend_movies(selected_movie_name, movies, similarity_matrix, top_n=10)
            
            if recommendations:
                st.success(f"Here are {len(recommendations)} movies similar to **{selected_movie_name}**:")
                st.markdown("---")
                
                # Display recommendations in a nice format
                for idx, movie in enumerate(recommendations, 1):
                    movie_info = movies[movies['title'] == movie].iloc[0]
                    with st.container():
                        st.markdown(f"**{idx}. {movie}**")
                        st.caption(f"Movie ID: {movie_info['movie_id']}")
                        st.markdown("---")
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("Please select a movie first!")

# Footer
st.markdown("---")
st.markdown("### About")
st.info("This recommendation system uses content-based filtering to suggest movies based on similar tags and descriptions. The more similar the content, the higher the recommendation score!")

