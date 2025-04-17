import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import re

def load_data():
    """
    Laddar in filmer, taggar och länkar från CSV-filer.
    
    Returns:
        movies_df: DataFrame med filmer
        tags_df: DataFrame med taggar
        links_df: DataFrame med länkar till IMDb och TMDB
    """
    # Läs in filmer
    movies_df = pd.read_csv("Data/movies.csv")
    
    # Läs in taggar (vi begränsar till de första 100,000 raderna för att spara minne)
    tags_df = pd.read_csv("Data/tags.csv", nrows=100000)
    
    # Konvertera taggar till strängar för att undvika TypeError
    tags_df['tag'] = tags_df['tag'].astype(str)

    # Läs in länkar till IMDb och TMDB
    links_df = pd.read_csv("Data/links.csv")
    
    return movies_df, tags_df, links_df

def combine_features(movies_df, tags_df):
    """
    Kombinerar genrer och taggar för att skapa innehållsegenskaper.
    
    Args:
        movies_df: DataFrame med filmer
        tags_df: DataFrame med taggar
    
    Returns:
        content_df: DataFrame med film-ID, titel, genrer och kombinerade egenskaper
    """
    # Gruppera taggar per film
    film_tags = tags_df.groupby('movieId')['tag'].apply(' '.join).reset_index()
    
    # Slå ihop filmer och taggar
    content_df = pd.merge(movies_df, film_tags, on='movieId', how='left')
    
    # Hantera saknade taggar
    content_df['tag'] = content_df['tag'].fillna('')
    
    # Kombinera genrer och taggar
    content_df['content'] = content_df['genres'].str.replace('|', ' ') + ' ' + content_df['tag']
    
    return content_df[['movieId', 'title', 'genres', 'content']]

def create_similarity_matrix(content_df):
    """
    Skapar en likhetsmatris baserad på innehållsegenskaperna.
    
    Args:
        content_df: DataFrame med innehållsegenskaper
    
    Returns:
        cosine_sim: Likhetsmatris
    """
    # Skapa TF-IDF-matris
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_df['content'])
    
    # Beräkna cosinus-likhet
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def get_recommendations(title, content_df, cosine_sim, num_recommendations=5):
    """
    Hämtar filmrekommendationer baserat på en given filmtitel.
    
    Args:
        title: Filmtitel att basera rekommendationer på
        content_df: DataFrame med innehållsegenskaper
        cosine_sim: Likhetsmatris
        num_recommendations: Antal rekommendationer att returnera
    
    Returns:
        recommendations: Lista med rekommenderade filmtitlar
    """
    # Hitta index för filmen
    idx = content_df[content_df['title'] == title].index
    
    if len(idx) == 0:
        return f"Filmen '{title}' hittades inte."
    
    idx = idx[0]
    
    # Hämta likhetspoäng
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sortera filmer efter likhet
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ta de mest lika filmerna (exklusive den givna filmen)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Hämta film-index
    movie_indices = [i[0] for i in sim_scores]
    
    # Returnera titlar
    return content_df['title'].iloc[movie_indices].tolist()

def tfidf_and_pca_vectors(content_df, use_pca, n_components):
    # Skapar TF-IDF-matris från innehållet och gör eventuell PCA-reduktion
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_df['content'])
    if use_pca:
        # Omvandla sparse-matris till dense oh reducera dimensioner med PCA
        tfidf_dense = tfidf_matrix.toarray()
        pca = PCA(n_components=min(n_components, tfidf_dense.shape[1], tfidf_dense.shape[0]))
        tfidf_matrix = pca.fit_transform(tfidf_dense)
    return tfidf_matrix

def calculate_similarity(selected_idx, tfidf_matrix):
    # Beräknar cosinuslikhet mellan vald film och alla andra filmer
    selected_vector = tfidf_matrix[selected_idx:selected_idx+1]
    cosine_sim = cosine_similarity(selected_vector, tfidf_matrix).flatten()
    return list(enumerate(cosine_sim))

def filter_out_similar_titles(sim_scores, content_df, movie_row, selected_movie):
    # Filtrerar bort filmer som troligen är från samma serie som den valda filmen
    selected_title = re.sub(r'\s*\(\d{4}\)\s*', '', selected_movie).strip()
    selected_genres = set(movie_row['genres'].iloc[0].split('|'))
    filtered = []
    for idx, score in sim_scores:
        title = re.sub(r'\s*\(\d{4}\)\s*', '', content_df['title'].iloc[idx]).strip()
        genres = set(content_df['genres'].iloc[idx].split('|'))
        # Hoppa över om titlarna är för lika
        if selected_title in title or title in selected_title:
            continue
        # Hoppa över om genrerna är identiska och första ordet i titeln är samma
        if genres == selected_genres and selected_title.split()[0] == title.split()[0]:
            continue
        filtered.append((idx, score))
    return filtered

def build_recommendation_list(movie_indices, content_df, links_df, cosine_sim):
    # Skapar en lista med rekommenderade filmer och deras info
    recommendations = []
    for idx in movie_indices:
        movie_id = content_df['movieId'].iloc[idx]
        link_row = links_df[links_df['movieId'] == movie_id]
        imdb_id = str(link_row['imdbId'].values[0]) if not link_row.empty and 'imdbId' in link_row.columns else None
        tmdb_id = str(link_row['tmdbId'].values[0]) if not link_row.empty and 'tmdbId' in link_row.columns else None
        recommendations.append({
            'title': content_df['title'].iloc[idx],
            'genres': content_df['genres'].iloc[idx],
            'imdb_id': imdb_id,
            'tmdb_id': tmdb_id,
            'similarity': cosine_sim[idx]
        })
    return recommendations

def get_movie_recommendations_optimized(selected_movie, content_df, links_df, num_recommendations=5, filter_similar_titles=True, use_pca=True, n_components=10):
    # Huvudfunktion som samordnar stegen för att skapa rekommendationer
    movie_row = content_df[content_df['title'] == selected_movie]
    if len(movie_row) == 0:
        return f"Filmen {selected_movie} hittades inte."
    selected_idx = movie_row.index[0]
    # Steg 1: Skapa TF-IDF och ev. PCA
    tfidf_matrix = tfidf_and_pca_vectors(content_df, use_pca, n_components)
    # Steg 2: Beräkna likheter
    sim_scores = calculate_similarity(selected_idx, tfidf_matrix)
    # Steg 3: Filtrera bort vald film själv
    sim_scores = [(idx, score) for idx, score in sim_scores if idx != selected_idx]
    # Steg 4: Filtrera bort filmer från samma serie om valt
    if filter_similar_titles:
        sim_scores = filter_out_similar_titles(sim_scores, content_df, movie_row, selected_movie)
    # Steg 5: Sortera och välja antal rekommendationer
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
    movie_indices = [i[0] for i in sim_scores]
    # Steg 6: Bygg och returnera rekommendationslistan
    return build_recommendation_list(movie_indices, content_df, links_df, dict(sim_scores))


    """
    Hämtar rekommendationer för en specifik film utan att beräkna hela likhetsmatrisen.
    Använder en kombination av innehållsbaserad filtrering och enkel titelfiltrering.
    Kan även använda PCA för att reducera dimensionalitet och förbättra rekommendationerna.
    
    Args:
        selected_movie: Filmtitel att basera rekommendationer på
        content_df: DataFrame med innehållsegenskaper
        links_df: DataFrame med länkar till IMDb och TMDB
        num_recommendations: Antal rekommendationer att returnera
        filter_similar_titles: Om True, försöker filtrera bort filmer från samma serie
        use_pca: Om True, använder PCA för dimensionalitetsreduktion
        n_components: Antal komponenter att använda i PCA
    
    Returns:
        recommendations: Lista med rekommenderade filmer och deras likhetspoäng samt länkar
    """


