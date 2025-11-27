import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

# TRAITEMENT DONNEES


def load_and_preprocess_data():
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    
    # Suppression des doublons de films basés sur le titre

    movies_df = movies_df.drop_duplicates(subset='movie_title', keep='first').reset_index(drop=True)

    
    # Fatures filtrage basé sur le contenu

    # Conversion des genres en  liste

    movies_df['genres'] = movies_df['genres'].str.split('|')
    
    # One-hot encoding pour les genres car par numérique

    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(movies_df['genres']),
                                columns=mlb.classes_,
                                index=movies_df.index)
    
    # Fusion des caractéristiques dans le  DataFrame des genres encodés
    content_features = pd.concat([
        genres_encoded,
        pd.get_dummies(movies_df['director_name']),
        pd.get_dummies(movies_df['actor_1_name']),
        pd.get_dummies(movies_df['actor_2_name']),
        pd.get_dummies(movies_df['actor_3_name'])
    ], axis=1)
    
    return ratings_df, movies_df, content_features



###########################################  FILTRAGE COLLABORATIF #################################################################################

def get_collaborative_recommendations(movie_title, movies_df, ratings_df, n_recommendations=3):

    #  l'ID du film sélectionné

    movie_id = movies_df[movies_df['movie_title'] == movie_title]['movieId'].iloc[0]
    
    # Matrice utilisateur-film

    ratings_matrix = pd.pivot_table(ratings_df, values='rating', 
                                  index='userId', columns='movieId', fill_value=0)
    
    
    movie_similarity = cosine_similarity(ratings_matrix.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, 
                                     index=ratings_matrix.columns,
                                     columns=ratings_matrix.columns)
    
    # Obtenir les films similaires
    similar_scores = movie_similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_scores.index[1:n_recommendations+1].tolist()
    
    recommendations = movies_df[movies_df['movieId'].isin(similar_movies)][['movie_title', 'genres']]
    return recommendations




###########################################  FILTRAGE BASE SUR LE CONTENU #################################################################################

def get_content_recommendations(movie_title, movies_df, content_features, n_recommendations=3):

    # Index du film sélectionné
    
    movie_idx = movies_df[movies_df['movie_title'] == movie_title].index[0]
    
    # Calculer la similarité basée sur le contenu
    similarity = cosine_similarity(content_features)
    similar_scores = pd.Series(similarity[movie_idx], index=movies_df.index)
    
    # Obtenir les films similaires
    similar_movie_indices = similar_scores.sort_values(ascending=False)[1:n_recommendations+1].index
    recommendations = movies_df.iloc[similar_movie_indices][['movie_title', 'genres']]
    return recommendations



def main():
    st.title("Movies recommandation system ")
    url = "https://www.linkedin.com/in/rimey-aboky-25603a20b/"
    github = "https://github.com/rimey2/movie_recommandations_appp.git"
    st.subheader(
        "This application allows you to have direct recommandations for your movies based on your liking"
    )
    st.sidebar.write("[Author : Rimey ABOKY](%s)" % url)
    st.sidebar.write("[Github : rimey2](%s)" % github)
    st.sidebar.markdown(
        "**To get started, follow the instructions below :** \n"
        "1. Check the movies in the list on the main page, \n"
        "1. Choose your movie , \n"
        "1. Press the recommandation button , \n"
        "1. Get your recommandations , \n"
    )
      
    ratings_df, movies_df, content_features = load_and_preprocess_data()
    
    # Sélection du film
    movie_titles = movies_df['movie_title'].unique()
    selected_movie = st.selectbox("Pick a movie :", movie_titles)
    
    if st.button("Get recommandations"):
        st.subheader("Recommandations based on collaborative filtering")
        try:
            collab_recommendations = get_collaborative_recommendations(selected_movie, movies_df, ratings_df)
            st.write(collab_recommendations)
        except Exception as e:
            st.write("Il n'y a pas assez de données d'évaluation pour ce film.")
        
        st.subheader("Recommandations based on content")
        content_recommendations = get_content_recommendations(selected_movie, movies_df, content_features)
        st.write(content_recommendations)

if __name__ == "__main__":
    main()