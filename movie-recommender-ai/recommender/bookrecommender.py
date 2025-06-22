from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

def is_likely_english(text):
    return bool(re.fullmatch(r"[ -~]+", text))  

class BookRecommender:
    def __init__(self, book_df):
        self.book_df = book_df.copy()

        self.book_df['Book'] = self.book_df['Book'].str.strip().str.lower()
        self.book_df['Author'] = self.book_df['Author'].str.strip().str.lower()

        self.book_df.drop_duplicates(subset=["Book", "Author"], inplace=True)

        self.book_df['Book'] = self.book_df['Book'].fillna('')
        self.book_df['Author'] = self.book_df['Author'].fillna('')
        self.book_df['Description'] = self.book_df['Description'].fillna('')
        self.book_df['Genres'] = self.book_df['Genres'].fillna('[]')
        self.book_df['book_key'] = self.book_df['Book'].str.strip().str.lower()
        self.book_df['author_key'] = self.book_df['Author'].str.strip().str.lower()


        self.book_df['Avg_Rating'] = pd.to_numeric(self.book_df['Avg_Rating'], errors='coerce')
        self.book_df['Num_Ratings'] = pd.to_numeric(
            self.book_df['Num_Ratings'].str.replace(',', ''), errors='coerce'
        )

        self.book_df.dropna(subset=['Avg_Rating', 'Num_Ratings'], inplace=True)

        def clean_genres(genre_str):
            try:
                genres = eval(genre_str) if isinstance(genre_str, str) else genre_str
                if isinstance(genres, list):
                    return ' '.join([str(g).lower().strip() for g in genres if isinstance(g, str)])
                return ''
            except:
                return ''

        self.book_df['cleaned_genres'] = self.book_df['Genres'].apply(clean_genres)

        self.book_df['text'] = (
            self.book_df['Book'] + ' ' +
            self.book_df['Author'] + ' ' +
            self.book_df['Description'] + ' ' +
            self.book_df['cleaned_genres']
        )

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.book_df['text'])

    def recommend(self, movie_title: str, genres=None, overview: str = "", mood: str = None, top_k: int = 5):
        genres = genres or []
        if not isinstance(genres, list):
            genres = [genres]

        genres_str = ' '.join([str(g).lower().strip() for g in genres if isinstance(g, str)])
        query_text = f"{movie_title.lower()} {genres_str} {overview.lower()}"
        query_vec = self.vectorizer.transform([query_text])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        self.book_df['similarity'] = sim_scores

        filtered = self.book_df[
            (self.book_df['Avg_Rating'] > 3.5) &
            (self.book_df['Num_Ratings'] > 1000)
        ].copy()

        if genres:
            input_genres = set([str(g).lower().strip() for g in genres])
            def genre_contains(row):
                try:
                    book_genres = eval(row['Genres']) if isinstance(row['Genres'], str) else row['Genres']
                    return any(str(bg).lower().strip() in input_genres for bg in book_genres if isinstance(bg, str))
                except:
                    return False
            filtered = filtered[filtered.apply(genre_contains, axis=1)]

        top_books = (
            filtered.sort_values(by='similarity', ascending=False)
            .drop_duplicates(subset=["book_key", "author_key"], keep='first')
        )

        top_books = top_books[top_books['Book'].apply(is_likely_english)].head(top_k)

        return top_books.drop(columns=['text', 'cleaned_genres', 'similarity', 'book_key', 'author_key'], errors='ignore').to_dict(orient='records')
