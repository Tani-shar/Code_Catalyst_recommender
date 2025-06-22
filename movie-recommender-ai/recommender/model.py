from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

from torch.nn.functional import cosine_similarity

from sklearn.cluster import KMeans
from fuzzywuzzy import process, fuzz
import pandas as pd
import numpy as np
import pickle
import os

class ContentRecommender:
    def __init__(self, df, cache_dir='cache', extras_path='data/imdb_top_1000.csv'):
        self.df = df
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.df['genres'] = self.df['genres'].fillna('').astype(str)
        self.df['startYear'] = pd.to_numeric(self.df['startYear'], errors='coerce').fillna(0).astype(int)
        self.df['runtimeMinutes'] = pd.to_numeric(self.df['runtimeMinutes'], errors='coerce').fillna(0).astype(float)
        self.df['isAdult'] = self.df['isAdult'].fillna(0).astype(int)
        self.df['titleType'] = self.df['titleType'].fillna('').astype(str)
        self.df['primaryTitle'] = self.df['primaryTitle'].fillna('').astype(str)
        self.df['originalTitle'] = self.df['originalTitle'].fillna('').astype(str)
        self.df['decade'] = (self.df['startYear'] // 10 * 10).astype(str).replace('.0', '')
        
        self.extras_lookup = {}
        print(f"📂 Looking for extras file at: {extras_path}")
        if os.path.exists(extras_path):
            extras_df = pd.read_csv(extras_path)
            # Clean keys for reliable matching
            extras_df['Series_Title'] = extras_df['Series_Title'].astype(str).str.strip().str.lower()
            extras_df['Released_Year'] = pd.to_numeric(extras_df['Released_Year'], errors='coerce').fillna(0).astype(int)
            
            for _, row in extras_df.iterrows():
                key = (row['Series_Title'], row['Released_Year'])
                self.extras_lookup[key] = {
                    "poster": row.get("Poster_Link", ""),
                    "overview": row.get("Overview", ""),
                    "certificate": row.get("Certificate", ""),
                    "director": row.get("Director", ""),
                    "stars": [s for s in [row.get(f"Star{i}") for i in range(1, 5)] if pd.notna(s)]
                }
            print(f"✅ Total extras loaded into lookup: {len(self.extras_lookup)}")
        else:
            print(f"❌ Extras file not found at path: {extras_path}")

        def get_movie_extras(row):
            title = row['primaryTitle'].lower().strip()
            year = row['startYear']
            # First, try a direct match
            data = self.extras_lookup.get((title, year))
            if data:
                return pd.Series([data.get('overview', ''), data.get('stars', [])])
            # If direct match fails, try a fuzzy match on title with the same year
            else:
                for (t, y), extras in self.extras_lookup.items():
                    if y == year and fuzz.ratio(t, title) >= 95:
                        return pd.Series([extras.get('overview', ''), extras.get('stars', [])])
            return pd.Series(['', []]) # Return empty if no match found

        print("⏳ Enriching DataFrame with overview and stars for model building...")
        self.df[['overview', 'stars']] = self.df.apply(get_movie_extras, axis=1)
        print(f"✅ Enrichment complete. {len(self.df[self.df['overview'] != ''])} movies have overviews.")
        self.df['has_extras'] = self.df['overview'].str.len() > 0
        print(f"✅ Flagged {self.df['has_extras'].sum()} movies with extra content.")

        def process_stars(star_list):
            if isinstance(star_list, list):
                return ' '.join([str(s).replace(' ', '').lower() for s in star_list])
            return ''

        self.df['processed_stars'] = self.df['stars'].apply(process_stars)
        
        self.df['combined'] = (
            self.df['genres'].str.replace(',', ' ') + ' ' + 
            self.df['decade'] + ' ' + 
            self.df['overview'] + ' ' +
            self.df['processed_stars']
        )
        
        self.titles = self.df['primaryTitle'].tolist()
        self.indices = pd.Series(self.df.index, index=self.df['primaryTitle']).drop_duplicates()

        tfidf_cache = os.path.join(cache_dir, 'tfidf_matrix_enriched.pkl')
        cos_sim_cache = os.path.join(cache_dir, 'cos_sim_enriched.pkl')

        if os.path.exists(tfidf_cache) and os.path.exists(cos_sim_cache):
            with open(tfidf_cache, 'rb') as f: self.tfidf_matrix = pickle.load(f)
            with open(cos_sim_cache, 'rb') as f: self.cos_sim = pickle.load(f)
            print("✅ Loaded ENRICHED TF-IDF and Cosine Similarity from cache.")
        else:
            print("⏳ Creating new ENRICHED TF-IDF and Cosine Similarity matrix...")
            model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, fast and accurate
            self.tfidf_matrix = model.encode(self.df['combined'].tolist(), convert_to_tensor=True, show_progress_bar=True)

            self.cos_sim = torch.matmul(self.tfidf_matrix, self.tfidf_matrix.T).cpu().numpy()

            with open(tfidf_cache, 'wb') as f: pickle.dump(self.tfidf_matrix, f)
            with open(cos_sim_cache, 'wb') as f: pickle.dump(self.cos_sim, f)
            print("✅ Saved new ENRICHED models to cache.")

        cluster_cache = os.path.join(cache_dir, 'clusters_enriched.pkl')
        if os.path.exists(cluster_cache):
            with open(cluster_cache, 'rb') as f: self.df['cluster'] = pickle.load(f)
        else:
            kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
            self.df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)
            with open(cluster_cache, 'wb') as f: pickle.dump(self.df['cluster'], f)

        def inject_extras(movie):
            title = str(movie.get('primaryTitle', '')).strip().lower()
            year = int(float(movie.get('startYear', 0)))

            best_score = 0
            best_key = None
            
            # First, check for a direct key match for speed
            if (title, year) in self.extras_lookup:
                best_key = (title, year)
                best_score = 100
            else: # Fallback to fuzzy search
                for (t, y), val in self.extras_lookup.items():
                    if y == year:
                        score = fuzz.ratio(t, title)
                        if score > best_score:
                            best_score = score
                            best_key = (t, y)

            if best_score >= 90:
                movie_extras = self.extras_lookup[best_key]
                movie['poster'] = movie_extras.get('poster', movie.get('poster', ''))
                movie['overview'] = movie_extras.get('overview', movie.get('overview', ''))
                movie['certificate'] = movie_extras.get('certificate', movie.get('certificate', ''))
                movie['stars'] = movie_extras.get('stars', movie.get('stars', []))
                movie['director'] = movie_extras.get('director', movie.get('director', ''))

            # Ensure essential keys always exist
            for key in ['poster', 'overview', 'certificate', 'director', 'stars', 'genres']:
                if key not in movie:
                    movie[key] = [] if key in ['stars', 'genres'] else ''

            return movie
            
        self.inject_extras = inject_extras

    def enrich_and_return(self, df):
        # We need to drop our helper columns before converting to dicts
        df_to_process = df.drop(columns=['overview', 'stars', 'processed_stars'], errors='ignore')
        records = df_to_process.to_dict(orient='records')
        return [self.inject_extras(m) for m in records]

    def adjust_for_profile(self, df, location=None, age=None):
        """
        Adjusts scores based on user profile (age) and adds a recency boost.
        This function now creates and returns the 'final_score'.
        """
        df_copy = df.copy()
        df_copy["boost"] = 1.0
        
        if age:
            try:
                age = int(age)
                if age < 12:
                    df_copy = df_copy[
                        ~df_copy["genres"].str.contains("Horror|Thriller|Romance|Crime|Mystery", na=False)
                    ]
                    df_copy = df_copy[df_copy["startYear"] >= 2010]

                elif age < 18:
                    df_copy = df_copy[~df_copy["genres"].str.contains("Horror|Thriller", na=False)]
                
                elif age < 30:
                    df_copy["boost"] *= df_copy["genres"].apply(
                        lambda g: 1.1 if "Romance" in g or "Action" in g else 1.0
                    )
            except (ValueError, TypeError):
                pass
        recency_weight = 0.2
        
        current_year = pd.to_datetime('today').year
        max_age = 30
        movie_age = current_year - df_copy['startYear']
        recency_score = 1 - (movie_age / max_age)
        recency_score = recency_score.clip(lower=0) 
        recency_boost = recency_score * recency_weight
        base_score = df_copy.get("score", 1.0)
        df_copy["final_score"] = (base_score + recency_boost) * df_copy["boost"]
        
        return df_copy.sort_values(by="final_score", ascending=False)

    def get_movie_suggestions(self, query, top_n=10):
        if not query or len(query.strip()) < 2: return []
        query = query.lower().strip()
        matches = process.extract(query, self.df['primaryTitle'].str.lower(), limit=top_n)
        suggestions = []
        added_ids = set()
        for m_title, score, index in matches:
            if score >= 70:
                row = self.df.iloc[index]
                if row['tconst'] not in added_ids:
                    suggestions.append({"title": row['primaryTitle'], "id": row['tconst'], "score": score})
                    added_ids.add(row['tconst'])
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)[:top_n]

    def _fallback(self, top_k, titleType, isAdult):
        df_filtered = self.df[self.df['genres'].str.contains('Comedy', case=False, na=False) & (self.df['isAdult'] == 0)]
        if titleType: df_filtered = df_filtered[df_filtered['titleType'] == titleType]
        if df_filtered.empty:
            df_filtered = self.df[self.df['isAdult'] == 0]
            if titleType: df_filtered = df_filtered[df_filtered['titleType'] == titleType]
        sort_col = 'averageRating' if 'averageRating' in df_filtered.columns else 'tconst'
        result = df_filtered.sort_values(by=sort_col, ascending=False).head(top_k).copy()
        result['reason'] = 'Popular fallback'
        return self.enrich_and_return(result)

    def _ensure_diversity(self, candidates, top_k):
        if len(candidates) <= top_k: return candidates
        selected = []
        genres_seen = set()
        clusters_seen = set()
        for _, row in candidates.iterrows():
            if len(selected) >= top_k: break
            primary_genre = row['genres'].split(',')[0] if ',' in row['genres'] else row['genres']
            cluster = row.get('cluster',-1)
            if (primary_genre not in genres_seen or cluster not in clusters_seen or len(selected) < top_k // 2):
                selected.append(_)
                genres_seen.add(primary_genre)
                clusters_seen.add(cluster)
        
        if len(selected) < top_k:
            remaining_indices = [idx for idx in candidates.index if idx not in selected]
            selected.extend(remaining_indices[:top_k-len(selected)])
            
        return candidates.loc[selected]

    def get_popular(self, top_k=50, titleType=None, isAdult=False):
        df_filtered = self.df.copy()
        if titleType: df_filtered = df_filtered[df_filtered['titleType'] == titleType]
        if not isAdult: df_filtered = df_filtered[df_filtered['isAdult'] == 0]
        sort_col = 'averageRating' if 'averageRating' in df_filtered.columns else 'tconst'
        result = df_filtered.sort_values(by=sort_col, ascending=False).head(top_k)
        result['genres'] = result['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
        result = result.fillna({'averageRating': 0, 'runtimeMinutes': 0, 'startYear': 0})
        return self.enrich_and_return(result)
    
    def recommend(self, movie_title, top_k=5, mood=None, timeofday=None, titleType=None, isAdult=False, location = None, age=None):
        titles_lower = {t.lower(): t for t in self.titles}
        match = process.extractOne(movie_title.lower(), titles_lower.keys())
        if not match or match[1] < 80: return self._fallback(top_k, titleType, isAdult)
        
        movie_title = titles_lower[match[0]]
        idx = self.indices[movie_title]
        sim_scores = list(enumerate(self.cos_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:(top_k*10)+1] # Get more candidates for filtering
        movie_indices = [i[0] for i in sim_scores]
        candidates = self.df.iloc[movie_indices].copy()
        candidates['score'] = [s[1] for s in sim_scores]
        candidates['reason'] = f'Similar to {movie_title}'
        
       
        candidates = self.adjust_for_profile(candidates, location=location,age=age)
        candidates = self._ensure_diversity(candidates.sort_values(by='final_score', ascending=False), top_k)
        result = candidates.head(top_k).copy()
        result['genres'] = result['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])

        return self.enrich_and_return(result)

    def recommend_for_user(self, liked_movies, ratings=None, mood=None, timeofday=None, titleType=None, isAdult=False, top_k=10,location= None, age=None):
        titles_lower = {t.lower(): t for t in self.titles}
        liked_indices, matched_titles = [], []
        for movie in liked_movies:
            match = process.extractOne(movie.lower(), titles_lower.keys())
            if match and match[1] >= 80:
                original_title = titles_lower[match[0]]
                liked_indices.append(self.indices[original_title])
                matched_titles.append(original_title)

        if not liked_indices: return self._fallback(top_k, titleType, isAdult)
        
        
        if ratings and len(ratings) == len(liked_indices):
            ratings_norm = np.array(ratings) / np.sum(ratings)
            mean_sim = np.average(self.cos_sim[liked_indices], axis=0, weights=ratings_norm)
        else:
            mean_sim = np.average(self.cos_sim[liked_indices], axis=0)

        candidates = self.df.copy()
        candidates['score'] = mean_sim
        candidates['reason'] = f'Similar to {", ".join(matched_titles)}'
        candidates = candidates[~candidates['primaryTitle'].isin(matched_titles)]
        extras_boost = 0.05  # A small value to act as a tie-breaker
        candidates.loc[candidates['has_extras'], 'score'] += extras_boost

        
        candidates = self.adjust_for_profile(candidates, location=location,age=age)
        candidates = self._ensure_diversity(candidates.sort_values(by='final_score', ascending=False), top_k)
        result = candidates.head(top_k).copy()
        result['genres'] = result['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
        # print(result[['primaryTitle', 'startYear', 'tconst']].head(10))
        # print(self.enrich_and_return(result))
        return self.enrich_and_return(result)
    
    def get_mood_based_movies(self, mood: str, timeofday: Optional[str] = None, top_k: int = 50,
                             titleType: Optional[str] = None, isAdult: bool = False) -> List[dict]:
        # Unchanged
        try:
            filtered_df = self.df[self.df['isAdult'] == isAdult]
            if titleType: filtered_df = filtered_df[filtered_df['titleType'] == titleType]
            mood_genre_map = { 'happy': ['Comedy', 'Family', 'Animation'], 'sad': ['Drama'], 'romantic': ['Romance'], 'thrilled': ['Thriller', 'Crime', 'Mystery']} # example
            if mood not in mood_genre_map: raise ValueError(f"Invalid mood: {mood}")
            mood_genres = mood_genre_map[mood]
            mood_relevant = filtered_df[filtered_df['genres'].str.contains('|'.join(mood_genres), na=False)]
            other_movies = filtered_df[~filtered_df.index.isin(mood_relevant.index)]
            n_mood, n_other = int(top_k * 0.7), int(top_k*0.3)
            mood_sample = mood_relevant.nlargest(n_mood, 'averageRating') if 'averageRating' in mood_relevant else mood_relevant.head(n_mood)
            other_sample = other_movies.nlargest(n_other, 'averageRating') if 'averageRating' in other_movies else other_movies.head(n_other)
            result = pd.concat([mood_sample, other_sample]).sample(frac=1).head(top_k)
            return self.enrich_and_return(result)
        except Exception as e:
            print(e)
            return self.get_popular(top_k=top_k, titleType=titleType, isAdult=isAdult)
