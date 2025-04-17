# Filmrekommendationssystem 🎬

Det här är ett filmrekommendationssystem som ger personliga filmtips baserat på innehållet i filmer – alltså deras genrer och användartaggar – med hjälp av MovieLens-datasetet.

Jag använde **Streamlit** för att skapa en enkel och användarvänlig webbapp där man kan söka efter en film och få rekommendationer som liknar filmen.

---

## 🚀 Funktioner

- Sök efter filmer
- Få 1–10 rekommendationer
- Filtrera bort filmer från samma serie för mer variation
- Se genre, likhetspoäng och länkar till IMDb och TMDB
- Optimerad för prestanda med caching

---

## 🧠 Hur systemet fungerar

### 📌 Innehållsbaserad filtrering  
Systemet föreslår filmer som liknar den du söker efter, baserat på filmens genre och taggar som andra användare har lagt till.

### 🧮 TF-IDF  
För att kunna jämföra filmer omvandlas text (som genrer och taggar) till siffror med TF-IDF. Det hjälper till att lyfta fram det som gör en film unik.

### 🧊 PCA + Filtrering för variation  
- **PCA** används för att förenkla datan så att systemet kan arbeta snabbare.
- En enkel **filtreringsalgoritm** ser till att inte alla rekommendationer kommer från samma filmserie.

### 📏 Cosinuslikhet  
Likheten mellan filmer beräknas med cosinuslikhet – det fokuserar på *vad* filmerna handlar om, inte hur mycket text som används.

---

## 🖥️ Användargränssnitt (Streamlit)

Webbappen låter dig:
- Söka på en film
- Välja hur många rekommendationer du vill ha (1–10)
- Slå på/av filtrering av filmserier
- Få länkar till IMDb och TMDB
- Se genrer och likhetspoäng

---

## ⚠️ Begränsningar

- **Datamängd:** Endast de första 100 000 taggarna används för att spara minne.
- **Nya filmer:** Om en film inte har genrer eller taggar, kan systemet inte rekommendera liknande. 
- **Ingen personlig historik:** Systemet lär sig inte vad en viss användare gillar.
- **Filtrering:** Den enkla filtreringen kan ibland missa eller ta bort fel filmer.
- **Språk:** Systemet är optimerat för engelskspråkiga filmer.
- **Beräkningseffektivitet:** PCA-transformationen kräver att den glesa TF-IDF-matrisen omvandlas till dense-format, vilket kan ta mycket minne vid stora datamängder.  
💡 *För att undvika prestandaproblem beräknas likheter endast mellan den valda filmen och övriga filmer – inte hela likhetsmatrisen. Det gör systemet mycket snabbare utan att försämra rekommendationerna märkbart.*

---

## 🛠️ Designval

- **Innehållsbaserat:** Rekommenderar filmer baserat på deras egenskaper (genrer, taggar) istället för användarnas betyg. Detta kräver ingen användardata och är enklare att bygga.
- **PCA med 10 komponenter:** Ger en bra balans mellan träffsäkerhet och prestanda.
- **Streamlit:** Gör det enkelt att snabbt skapa ett snyggt gränssnitt.
- **Optimerad beräkning:** Jämför bara den valda filmen med andra, vilket sparar mycket tid.

---

## 📦 Installation

1. Klona eller ladda ner projektet
2. Installera nödvändiga paket:
```bash
pip install pandas numpy scikit-learn streamlit
```

3. Ladda ner MovieLens-datasetet och placera filerna i en mapp kallad Data
4. Starta applikationen:
```bash
streamlit run streamlit_app.py
```

---

## 📚 Källhänvisningar

- **MovieLens Dataset** – [https://grouplens.org/datasets/movielens/]
  Används för filmdata, genrer och användartaggar.

- **scikit-learn** – [https://scikit-learn.org/] 
  Används för TF-IDF-vektorisering, PCA och cosinuslikhet.

- **Streamlit** – [https://streamlit.io/]
  Används för att bygga gränssnittet för webbapplikationen.

- **Content-Based Filtering** – [https://developers.google.com/machine-learning/recommendation/content-based/basics] 
  Metodiken för innehållsbaserad filtrering som används i systemet.

- **TF-IDF Vectorization** – [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html]
  Dokumentation för TF-IDF-vektorisering i scikit-learn.

- **Principal Component Analysis (PCA)** – [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html]
  Dokumentation för PCA-implementationen i scikit-learn.

- **Cosine Similarity** – [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html] 
  Dokumentation för cosinuslikhet i scikit-learn.

- **Övriga verktyg** - Tagit hjälp av Stack Overflow och ChatGPT för lite syntax och felsökning.


