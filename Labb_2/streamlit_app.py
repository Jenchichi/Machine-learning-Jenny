import streamlit as st
from Laboration2_film_recommender import load_data, combine_features, get_movie_recommendations_optimized

# Konfigurera Streamlit-sidan
st.set_page_config(
    page_title="Filmrekommendationssystem",
    page_icon="🎬"
)

# Titel och beskrivning
st.title("Filmrekommendationssystem")
st.write("Välj en film för att få rekommendationer på liknande filmer baserat på genrer och taggar.")

# Ladda data (med cache för att undvika att ladda om data varje gång)
@st.cache_data
def load_app_data():
    with st.spinner("Laddar data..."):
        movies_df, tags_df, links_df = load_data()
        content_df = combine_features(movies_df, tags_df)
        return movies_df, content_df, links_df

# Ladda data
movies_df, content_df, links_df = load_app_data()

# Skapa en sökruta där användaren kan skriva in filmtitel
selected_movie = st.text_input("Skriv in en filmtitel:", "")

# Visa förslag när användaren börjar skriva
if selected_movie:
    # Filtrera filmer baserat på användarens inmatning
    matching_movies = movies_df[movies_df['title'].str.contains(selected_movie, case=False)]['title'].tolist()
    
    if matching_movies:
        selected_movie = st.selectbox("Välj från matchande filmer:", matching_movies)
    else:
        st.warning(f"Inga filmer matchade '{selected_movie}'. Försök med en annan titel.")
        selected_movie = None

# Antal rekommendationer
num_recommendations = st.slider(
    "Antal rekommendationer:", 
    min_value=1, 
    max_value=10, 
    value=5
)

# Lägg till en checkbox för att filtrera bort filmer från samma serie och använda PCA
filter_similar = st.checkbox(
    "Filtrera bort filmer från samma serie", 
    value=True,
    help="Om markerad, använder systemet PCA för att filtrera bort uppföljare och filmer i samma serie"
)

# Använd PCA om filtreringen är aktiverad
use_pca = filter_similar

# Använder fast värde för PCA-komponenter (10 är optimalt efter test)
n_components = 10  

# Knapp för att generera rekommendationer
if st.button("Ge mig rekommendationer"):
    if selected_movie:
        with st.spinner("Beräknar rekommendationer..."):
            # Hämta rekommendationer med den optimerade funktionen
            recommendations = get_movie_recommendations_optimized(
                selected_movie, 
                content_df,
                links_df, 
                num_recommendations=num_recommendations,
                filter_similar_titles=filter_similar,
                use_pca=use_pca,
                n_components=n_components
            )
        
        # Visa rekommendationer
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.subheader(f"Baserat på '{selected_movie}', rekommenderar vi:")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. **{rec['title']}**")
                st.write(f"   Genrer: {rec['genres']}")
                st.write(f"   Likhet: {rec['similarity']:.2f}")

                # Skapa länkar till TMDB om ID:n finns
                if rec['tmdb_id'] is not None:
                    tmdb_url = f"https://www.themoviedb.org/movie/{rec['tmdb_id']}"
                    st.write(f"   [🎥 TMDB]({tmdb_url})")

                st.write("---")
    else:
        st.warning("Välj en film först!")

# Visa information om projektet
with st.expander("Om projektet"):
    st.write("""
    Detta filmrekommendationssystem hjälper dig hitta filmer som liknar den du gillar.

    🔍 Hur det funkar:
    Systemet analyserar både filmens genrer och taggar som användare lagt till. På så sätt får varje film en unik "profil" som gör det lättare att hitta liknande filmer.
    
    Systemet jämför sedan din valda film med andra filmer och visar de som är mest lika.
    
    För att ge dig mer varierade rekommendationer går det att automatiskt filtrera bort filmer från samma serie.

    Tekniker som används:
    - TF-IDF (Term Frequency-Inverse Document Frequency) för att omvandla text till vektorer
    - Cosinus-likhet för att mäta likhet mellan filmer
    - Optimerad beräkning som bara jämför den valda filmen med andra filmer
    
    Datakälla: MovieLens dataset (ml-latest)
    """)
