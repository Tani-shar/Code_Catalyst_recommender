from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

def is_likely_english(text):
    return bool(re.fullmatch(r"[ -~]+", text))  

class SongRecommender:
    def __init__(self, song_df):
        self.song_df = song_df.copy()

        # Clean up NaNs
        self.song_df['track_name'] = self.song_df['track_name'].fillna('')
        self.song_df['album_name'] = self.song_df['album_name'].fillna('')
        self.song_df['artists'] = self.song_df['artists'].fillna('')
        self.song_df['track_genre'] = self.song_df['track_genre'].fillna('')

        # Combine textual fields
        self.song_df['text'] = (
            self.song_df['track_name'].str.lower() + ' ' +
            self.song_df['album_name'].str.lower() + ' ' +
            self.song_df['artists'].str.lower() + ' ' +
            self.song_df['track_genre'].str.lower()
        )

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.song_df['text'])

    def recommend(self, movie_title, genres=None, overview="", mood=None, top_k=5):
        genres = genres or []
        combined_text = movie_title.lower() + " " + " ".join(genres).lower() + " " + overview.lower()
        query_vec = self.vectorizer.transform([combined_text])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply mood filters
        valence_range = (0.2, 0.9)
        danceability_range = (0.3, 0.9)

        if mood:
            mood_profiles = {
                "happy": ((0.6, 1.0), (0.5, 1.0)),
                "sad": ((0.0, 0.4), (0.2, 0.6)),
                "thoughtful": ((0.2, 0.5), (0.3, 0.6)),
                "energetic": ((0.6, 1.0), (0.6, 1.0)),
                "romantic": ((0.4, 0.7), (0.4, 0.7))
            }
            if mood in mood_profiles:
                valence_range, danceability_range = mood_profiles[mood]

        self.song_df["similarity"] = sim_scores
        filtered = self.song_df[
            self.song_df['valence'].between(*valence_range) &
            self.song_df['danceability'].between(*danceability_range)
        ]

        top_recommendations = filtered.sort_values(by="similarity", ascending=False)
        top_recommendations = top_recommendations[top_recommendations['track_name'].apply(is_likely_english)].head(top_k)

        return top_recommendations.drop(columns=["text", "similarity"]).to_dict(orient="records")
