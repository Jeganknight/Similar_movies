import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from imdb import IMDb

# Initialize IMDb
ia = IMDb()

# Streamlit UI
st.title("🎥 Similar Movie Recommendation Assistant")
st.subheader("Step 1: Enter API Key")

# User input for API key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your API key to proceed.")
    st.stop()

# Initialize Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Initialize the prompt templates
synopsis_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable movie recommendation assistant specializing in global cinema.
    Based on the user's provided movie's synopsis and genres, find 10 similar movies from the entire IMDb database,
    including their titles, years, countries of origin, storylines, IMDb ratings, and explain why each of the movies is similar to the provided movie.

    Movie Synopsis:
    {storyline}

    Movie IMDb Rating:
    {rating}

    Movie Genres:
    {genres}

    Your Response:
    Provide a list of 10 movies with their titles, years, countries, IMDb ratings, storylines, and explain the similarity of each movie with the provided movie's storyline.
    """
)

plot_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable movie recommendation assistant specializing in global cinema.
    Based on the user's provided movie's plot and genres, find 3 similar movies from the entire IMDb database,
    including their titles, years, countries of origin, storylines, IMDb ratings, and explain why each of the movies is similar to the provided movie.

    Movie Plot:
    {storyline}

    Movie IMDb Rating:
    {rating}

    Movie Genres:
    {genres}

    Your Response:
    Provide a list of 3 movies with their titles, years, countries, IMDb ratings, storylines, and explain the similarity of each movie with the provided movie's storyline.
    """
)

st.subheader("Step 2: Provide Movie Details")
movie_name = st.text_input("Enter Movie Name")
movie_year = st.number_input("Enter Movie Year", min_value=1900, max_value=2024, step=1)

if st.button("Get Similar Movie Recommendations"):
    if not movie_name or not movie_year:
        st.error("Please provide both movie name and year.")
        st.stop()

    # Search IMDb for the movie by name and year
    search_results = ia.search_movie(movie_name)
    matching_movie = None

    # Find the movie with the exact match for year
    for movie in search_results:
        if movie.get('year') == movie_year:
            matching_movie = ia.get_movie(movie.movieID)
            break

    if not matching_movie:
        st.error("No matching movie found for the given name and year.")
        st.stop()

    # Fetch the movie's synopsis or plot and IMDb rating
    movie_synopsis = matching_movie.get('synopsis')
    movie_plot = matching_movie.get('plot')
    movie_rating = matching_movie.get('rating')
    movie_genres = ", ".join(matching_movie.get('genres', []))

    if movie_synopsis:
        user_input_synopsis = {
            "storyline": movie_synopsis,
            "rating": movie_rating or "Not available",
            "genres": movie_genres,
        }
        prompt_synopsis = synopsis_prompt_template.format(**user_input_synopsis)

        try:
            response_synopsis = llm.invoke(prompt_synopsis)
            st.subheader("Similar Movie Recommendations (Based on Synopsis):")
            st.write(response_synopsis.content.strip())
        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif movie_plot:
        user_input_plot = {
            "storyline": movie_plot[0],
            "rating": movie_rating or "Not available",
            "genres": movie_genres,
        }
        prompt_plot = plot_prompt_template.format(**user_input_plot)

        try:
            response_plot = llm.invoke(prompt_plot)
            st.subheader("Similar Movie Recommendations (Based on Plot):")
            st.write(response_plot.content.strip())
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("No storyline or plot available for the selected movie.")
