pip install pandas numpy implicit scipy scikit-learn joblib requests lightfm implicit optuna optuna-integration[tfkeras]

import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HybridRecommenderV3:
    """
    An improved Hybrid Movie Recommendation Engine with comprehensive fixes:
    - Fixed index alignment issues
    - Added robust error handling and validation
    - Improved memory efficiency
    - Added logging and monitoring
    - Enhanced cold start handling
    - Added model versioning and metadata
    """

    def __init__(
        self,
        data_paths: Dict[str, str],
        artifacts_path: str = './recommender_artifacts_v3',
        als_factors: int = 100,
        als_iterations: int = 25,
        als_regularization: float = 0.01,
        weights: Dict[str, float] = None,
        age_split_young: int = 25,
        age_split_old: int = 45,
        cb_rating_threshold: float = 4.0,
        min_ratings_per_user: int = 5,
        min_ratings_per_movie: int = 10,
        enable_logging: bool = True
    ):
        """Initialize the hybrid recommender with enhanced configuration and validation."""

        # Setup logging
        if enable_logging:
            self._setup_logging()

        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Initializing Hybrid Recommender V3 ===")

        # Validate inputs
        self._validate_inputs(data_paths, weights)

        # Configuration with validation
        self.config = {
            'data_paths': data_paths,
            'artifacts_path': Path(artifacts_path),
            'als_factors': max(10, min(500, als_factors)),  # Constrain factors
            'als_iterations': max(5, min(100, als_iterations)),  # Constrain iterations
            'als_regularization': max(0.001, min(1.0, als_regularization)),
            'weights': weights or {'cf': 0.4, 'cb': 0.4, 'context': 0.2},
            'age_splits': {'young': age_split_young, 'old': age_split_old},
            'cb_rating_threshold': max(1.0, min(5.0, cb_rating_threshold)),
            'min_ratings_per_user': max(1, min_ratings_per_user),
            'min_ratings_per_movie': max(1, min_ratings_per_movie),
            'model_version': '3.0',
            'created_at': datetime.now().isoformat()
        }

        # Initialize artifacts storage
        self.artifacts = {}
        self.is_trained = False

        # Validate weight distribution
        total_weight = sum(self.config['weights'].values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            self.config['weights'] = {k: v/total_weight for k, v in self.config['weights'].items()}

        # Load or train model
        if not self._load_artifacts():
            self.logger.info("Could not load artifacts. Training from scratch...")
            try:
                self._train_and_prepare_artifacts()
                self._save_artifacts()
                self.is_trained = True
                self.logger.info("Training completed successfully.")
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                raise
        else:
            self.is_trained = True
            self.logger.info("Artifacts loaded successfully from cache.")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )

    def _validate_inputs(self, data_paths: Dict[str, str], weights: Optional[Dict[str, float]]):
        """Validate input parameters."""
        required_files = ['users', 'movies', 'ratings']

        if not all(key in data_paths for key in required_files):
            raise ValueError(f"data_paths must contain keys: {required_files}")

        for key, path in data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

        if weights:
            if not all(isinstance(v, (int, float)) and v >= 0 for v in weights.values()):
                raise ValueError("All weights must be non-negative numbers")

    def _load_data(self) -> bool:
        """Load and validate data from source files."""
        self.logger.info("Loading data from source files...")

        try:
            # Load data with proper error handling
            users = pd.read_csv(
                self.config['data_paths']['users'],
                sep='::',
                engine='python',
                names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                encoding='latin-1'
            )

            ratings = pd.read_csv(
                self.config['data_paths']['ratings'],
                sep='::',
                engine='python',
                names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                encoding='latin-1'
            )

            movies = pd.read_csv(
                self.config['data_paths']['movies'],
                sep='::',
                engine='python',
                names=['MovieID', 'Title', 'Genres'],
                encoding='latin-1'
            )

            # Validate data integrity
            if not self._validate_data(users, ratings, movies):
                return False

            # Filter data based on minimum requirements
            ratings = self._filter_sparse_data(ratings)

            # Store clean data
            self.artifacts['users'] = users.set_index('UserID')
            self.artifacts['movies'] = movies.set_index('MovieID')
            self.artifacts['ratings'] = ratings

            self.logger.info(f"Data loaded: {len(users)} users, {len(movies)} movies, {len(ratings)} ratings")
            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False

    def _validate_data(self, users: pd.DataFrame, ratings: pd.DataFrame, movies: pd.DataFrame) -> bool:
        """Validate data integrity and consistency."""

        # Check for empty dataframes
        if any(df.empty for df in [users, ratings, movies]):
            self.logger.error("One or more dataframes are empty")
            return False

        # Check for required columns
        if not all(col in users.columns for col in ['UserID', 'Gender', 'Age']):
            self.logger.error("Missing required columns in users data")
            return False

        if not all(col in ratings.columns for col in ['UserID', 'MovieID', 'Rating']):
            self.logger.error("Missing required columns in ratings data")
            return False

        if not all(col in movies.columns for col in ['MovieID', 'Title', 'Genres']):
            self.logger.error("Missing required columns in movies data")
            return False

        # Check for data consistency
        rating_users = set(ratings['UserID'].unique())
        rating_movies = set(ratings['MovieID'].unique())

        available_users = set(users['UserID'].unique())
        available_movies = set(movies['MovieID'].unique())

        missing_users = rating_users - available_users
        missing_movies = rating_movies - available_movies

        if missing_users:
            self.logger.warning(f"Found {len(missing_users)} users in ratings without user data")

        if missing_movies:
            self.logger.warning(f"Found {len(missing_movies)} movies in ratings without movie data")

        # Check rating scale
        rating_range = ratings['Rating'].min(), ratings['Rating'].max()
        if rating_range[0] < 1 or rating_range[1] > 5:
            self.logger.warning(f"Unusual rating range: {rating_range}")

        return True

    def _filter_sparse_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filter out users and movies with insufficient ratings."""
        original_size = len(ratings)

        # Filter users with minimum ratings
        user_counts = ratings['UserID'].value_counts()
        valid_users = user_counts[user_counts >= self.config['min_ratings_per_user']].index
        ratings = ratings[ratings['UserID'].isin(valid_users)]

        # Filter movies with minimum ratings
        movie_counts = ratings['MovieID'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.config['min_ratings_per_movie']].index
        ratings = ratings[ratings['MovieID'].isin(valid_movies)]

        filtered_size = len(ratings)
        self.logger.info(f"Filtered ratings: {original_size} -> {filtered_size} ({filtered_size/original_size:.1%} retained)")

        return ratings

    def _train_and_prepare_artifacts(self):
        """Train models and prepare all artifacts with improved error handling."""
        if not self._load_data():
            raise RuntimeError("Failed to load data")

        self.logger.info("Creating user and movie mappings...")
        self._create_mappings()

        self.logger.info("Training Collaborative Filtering model...")
        self._train_collaborative_filtering()

        self.logger.info("Building Content-Based model...")
        self._train_content_based()

        self.logger.info("Pre-computing user profiles...")
        self._build_all_cb_user_profiles()

        self.logger.info("Loading cross-domain data...")
        self.load_cross_domain_data()

        self.logger.info("Artifact preparation complete.")

    def _create_mappings(self):
        """Create consistent mappings between IDs and indices."""
        users_df = self.artifacts['users'].reset_index()
        movies_df = self.artifacts['movies'].reset_index()

        # Create sorted mappings for consistency
        unique_users = sorted(users_df['UserID'].unique())
        unique_movies = sorted(movies_df['MovieID'].unique())

        self.artifacts['user_map'] = {uid: i for i, uid in enumerate(unique_users)}
        self.artifacts['movie_map'] = {mid: i for i, mid in enumerate(unique_movies)}
        self.artifacts['user_map_inv'] = {i: uid for uid, i in self.artifacts['user_map'].items()}
        self.artifacts['movie_map_inv'] = {i: mid for mid, i in self.artifacts['movie_map'].items()}

        self.logger.info(f"Created mappings: {len(unique_users)} users, {len(unique_movies)} movies")

    def _train_collaborative_filtering(self):
        """Train collaborative filtering model with proper error handling."""

        ratings = self.artifacts['ratings'].copy()

        # Map to internal indices
        ratings['user_idx'] = ratings['UserID'].map(self.artifacts['user_map'])
        ratings['movie_idx'] = ratings['MovieID'].map(self.artifacts['movie_map'])

        # Remove unmapped entries
        ratings = ratings.dropna(subset=['user_idx', 'movie_idx'])
        ratings['user_idx'] = ratings['user_idx'].astype(int)
        ratings['movie_idx'] = ratings['movie_idx'].astype(int)

        if ratings.empty:
            raise RuntimeError("No valid ratings after mapping")

        # Create user-item matrix with confidence weights
        n_users = len(self.artifacts['user_map'])
        n_movies = len(self.artifacts['movie_map'])

        # Add confidence weights based on rating values
        ratings['confidence'] = 1 + ratings['Rating'] * 0.5  # Higher ratings get more weight

        user_item_matrix = csr_matrix(
            (ratings['confidence'].values, (ratings['user_idx'].values, ratings['movie_idx'].values)),
            shape=(n_users, n_movies)
        )

        # Train ALS model with improved parameters
        cf_model = AlternatingLeastSquares(
            factors=self.config['als_factors'],
            regularization=self.config['als_regularization'],
            iterations=self.config['als_iterations'],
            random_state=42,
            use_gpu=False,  # Ensure CPU usage for compatibility
            calculate_training_loss=True,  # Monitor training progress
            alpha=40.0  # Increased alpha for better implicit feedback handling
        )

        # Fit on transposed matrix (implicit expects items x users)
        cf_model.fit(user_item_matrix.T)

        self.artifacts['cf_model'] = cf_model
        self.artifacts['user_item_matrix'] = user_item_matrix

        self.logger.info(f"CF model trained: {n_users} users, {n_movies} movies, density: {user_item_matrix.nnz/(n_users*n_movies):.4f}")

    def _train_content_based(self):
        """Train content-based model with improved preprocessing."""

        movies_copy = self.artifacts['movies'].copy()

        # Enhanced text preprocessing
        movies_copy['Genres_processed'] = movies_copy['Genres'].str.replace('|', ' ').str.lower()

        # Extract year from title for temporal features
        movies_copy['Year'] = movies_copy['Title'].str.extract(r'\((\d{4})\)')
        movies_copy['Year'] = pd.to_numeric(movies_copy['Year'], errors='coerce')

        # Create decade feature
        movies_copy['Decade'] = (movies_copy['Year'] // 10) * 10

        # Handle missing values
        movies_copy['Genres_processed'] = movies_copy['Genres_processed'].fillna('unknown')
        movies_copy['Year'] = movies_copy['Year'].fillna(movies_copy['Year'].median())
        movies_copy['Decade'] = movies_copy['Decade'].fillna(movies_copy['Decade'].median())

        # Create TF-IDF matrix with improved parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=2000,  # Increased features for better representation
            min_df=2,  # Ignore very rare terms
            max_df=0.95,  # Ignore very common terms
            ngram_range=(1, 3),  # Include trigrams for better context
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )

        try:
            # Create genre-based TF-IDF matrix
            tfidf_matrix = tfidf.fit_transform(movies_copy['Genres_processed'])

            # Add temporal features
            year_scaler = MinMaxScaler()
            decade_scaler = MinMaxScaler()

            year_features = year_scaler.fit_transform(movies_copy['Year'].values.reshape(-1, 1))
            decade_features = decade_scaler.fit_transform(movies_copy['Decade'].values.reshape(-1, 1))

            # Combine features
            combined_features = np.hstack([
                tfidf_matrix.toarray(),
                year_features,
                decade_features
            ])

            # Store all artifacts
            self.artifacts['tfidf_vectorizer'] = tfidf
            self.artifacts['tfidf_matrix'] = csr_matrix(combined_features)
            self.artifacts['year_scaler'] = year_scaler
            self.artifacts['decade_scaler'] = decade_scaler
            self.artifacts['movies_processed'] = movies_copy

            self.logger.info(f"TF-IDF matrix created: {tfidf_matrix.shape}")

        except Exception as e:
            self.logger.error(f"Error creating TF-IDF matrix: {e}")
            raise

    def _build_all_cb_user_profiles(self):
        """Build content-based user profiles with improved efficiency."""

        user_profiles = {}
        ratings = self.artifacts['ratings']
        tfidf_matrix = self.artifacts['tfidf_matrix']
        movie_map = self.artifacts['movie_map']

        # Create global fallback profile
        global_profile = tfidf_matrix.mean(axis=0)

        # Process users in batches for memory efficiency
        batch_size = 1000
        user_ids = list(self.artifacts['users'].index)

        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]

            for user_id in batch_users:
                try:
                    user_rated_movies = ratings[ratings['UserID'] == user_id]

                    if user_rated_movies.empty:
                        user_profiles[user_id] = global_profile
                        continue

                    # Focus on highly rated movies
                    highly_rated = user_rated_movies[
                        user_rated_movies['Rating'] >= self.config['cb_rating_threshold']
                    ]

                    if not highly_rated.empty:
                        # Get valid movie indices
                        rated_indices = [
                            movie_map[mid] for mid in highly_rated['MovieID']
                            if mid in movie_map
                        ]

                        if rated_indices:
                            # Weight by rating
                            weights = highly_rated[highly_rated['MovieID'].isin([
                                self.artifacts['movie_map_inv'][idx] for idx in rated_indices
                            ])]['Rating'].values

                            # Normalize weights
                            weights = weights / weights.sum()

                            # Create weighted profile
                            weighted_profiles = tfidf_matrix[rated_indices].multiply(weights.reshape(-1, 1))
                            user_profiles[user_id] = weighted_profiles.sum(axis=0)
                        else:
                            user_profiles[user_id] = global_profile
                    else:
                        # Use all ratings if no highly rated movies
                        all_rated_indices = [
                            movie_map[mid] for mid in user_rated_movies['MovieID']
                            if mid in movie_map
                        ]

                        if all_rated_indices:
                            user_profiles[user_id] = tfidf_matrix[all_rated_indices].mean(axis=0)
                        else:
                            user_profiles[user_id] = global_profile

                except Exception as e:
                    self.logger.warning(f"Error processing user {user_id}: {e}")
                    user_profiles[user_id] = global_profile

        self.artifacts['user_cb_profiles'] = user_profiles
        self.artifacts['global_cb_profile'] = global_profile

        self.logger.info(f"Built {len(user_profiles)} user profiles")

    def _get_context_scores(self, user_id: int, candidate_movie_ids: pd.Index) -> pd.Series:
        """Generate context-aware scores based on user demographics and movie genres."""

        try:
            user_data = self.artifacts['users'].loc[user_id]
            user_age = user_data['Age']
            user_gender = user_data['Gender']
            user_occupation = user_data['Occupation']

            candidate_movies = self.artifacts['movies'].loc[candidate_movie_ids]
            movies_processed = self.artifacts['movies_processed']

            scores = {}
            age_young, age_old = self.config['age_splits']['young'], self.config['age_splits']['old']

            # Define age-genre preferences with weights
            age_genre_preferences = {
                'young': {
                    'Action': 1.0, 'Comedy': 0.9, 'Sci-Fi': 0.9, 'Adventure': 0.8,
                    'Fantasy': 0.8, 'Animation': 0.7, 'Horror': 0.7, 'Thriller': 0.6
                },
                'mature': {
                    'Drama': 1.0, 'Romance': 0.9, 'War': 0.8, 'Documentary': 0.8,
                    'Film-Noir': 0.7, 'Biography': 0.7, 'History': 0.6, 'Crime': 0.6
                }
            }

            # Define gender-genre preferences
            gender_genre_preferences = {
                'M': {'Action': 0.8, 'War': 0.7, 'Sci-Fi': 0.7, 'Crime': 0.6},
                'F': {'Romance': 0.8, 'Drama': 0.7, 'Musical': 0.7, 'Comedy': 0.6}
            }

            # Define occupation-genre preferences
            occupation_genre_preferences = {
                0: {'Documentary': 0.8, 'Drama': 0.7},  # Other
                1: {'Action': 0.8, 'Adventure': 0.7},   # Academic/Educator
                2: {'Drama': 0.8, 'Romance': 0.7},     # Artist
                3: {'Comedy': 0.8, 'Action': 0.7},     # Clerical/Admin
                4: {'Drama': 0.8, 'Thriller': 0.7},    # College/Grad Student
                5: {'Action': 0.8, 'Sci-Fi': 0.7},     # Customer Service
                6: {'Drama': 0.8, 'Comedy': 0.7},      # Doctor/Health Care
                7: {'Action': 0.8, 'Adventure': 0.7},  # Executive/Managerial
                8: {'Drama': 0.8, 'Romance': 0.7},     # Farmer
                9: {'Action': 0.8, 'Comedy': 0.7},     # Homemaker
                10: {'Drama': 0.8, 'Comedy': 0.7},     # K-12 Student
                11: {'Action': 0.8, 'Adventure': 0.7}, # Lawyer
                12: {'Drama': 0.8, 'Comedy': 0.7},     # Programmer
                13: {'Action': 0.8, 'Sci-Fi': 0.7},    # Retired
                14: {'Drama': 0.8, 'Romance': 0.7},    # Sales/Marketing
                15: {'Action': 0.8, 'Adventure': 0.7}, # Scientist
                16: {'Drama': 0.8, 'Comedy': 0.7},     # Self-employed
                17: {'Action': 0.8, 'Comedy': 0.7},    # Technician/Engineer
                18: {'Drama': 0.8, 'Romance': 0.7},    # Tradesman/Craftsman
                19: {'Action': 0.8, 'Adventure': 0.7}, # Unemployed
                20: {'Drama': 0.8, 'Comedy': 0.7}      # Writer
            }

            for movie_id, movie in candidate_movies.iterrows():
                score = 0.0
                genres = set(movie['Genres'].split('|'))

                # Age-based scoring
                if user_age <= age_young:
                    age_prefs = age_genre_preferences['young']
                elif user_age >= age_old:
                    age_prefs = age_genre_preferences['mature']
                else:
                    # Middle age - combine both preferences
                    age_prefs = {**age_genre_preferences['young'], **age_genre_preferences['mature']}

                age_score = sum(age_prefs.get(genre, 0.5) for genre in genres) / len(genres) if genres else 0

                # Gender-based scoring
                gender_prefs = gender_genre_preferences.get(user_gender, {})
                gender_score = sum(gender_prefs.get(genre, 0.5) for genre in genres) / len(genres) if genres else 0

                # Occupation-based scoring
                occupation_prefs = occupation_genre_preferences.get(user_occupation, {})
                occupation_score = sum(occupation_prefs.get(genre, 0.5) for genre in genres) / len(genres) if genres else 0

                # Temporal scoring (preference for movies from user's era)
                movie_year = movies_processed.loc[movie_id, 'Year']
                year_diff = abs(movie_year - (2024 - user_age))  # Approximate user's birth year
                temporal_score = 1.0 / (1.0 + year_diff / 10.0)  # Decay factor of 10 years

                # Combine scores with weights
                score = (
                    0.4 * age_score +
                    0.3 * gender_score +
                    0.2 * occupation_score +
                    0.1 * temporal_score
                )

                scores[movie_id] = score

            return pd.Series(scores, name='context_score').reindex(candidate_movie_ids).fillna(0.0)

        except Exception as e:
            self.logger.warning(f"Error computing context scores: {e}")
            return pd.Series(0.0, index=candidate_movie_ids, name='context_score')

    def get_recommendations(self, user_id: int, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Generate personalized movie recommendations with comprehensive error handling.

        Args:
            user_id: Target user ID
            top_n: Number of recommendations to return

        Returns:
            DataFrame with recommendations or None if error
        """

        if not self.is_trained:
            self.logger.error("Model is not trained. Cannot generate recommendations.")
            return None

        if user_id not in self.artifacts['user_map']:
            self.logger.warning(f"User {user_id} not found. Attempting cold start recommendation...")
            return self._cold_start_recommendations(user_id, top_n)

        self.logger.info(f"Generating recommendations for UserID: {user_id}")

        try:
            user_idx = self.artifacts['user_map'][user_id]

            # Create recommendation DataFrame
            available_movies = self.artifacts['movies'].index
            rec_df = pd.DataFrame(index=available_movies)

            # 1. Collaborative Filtering Scores
            cf_scores = self._get_cf_scores(user_idx)
            rec_df['cf_score'] = cf_scores

            # 2. Content-Based Scores
            cb_scores = self._get_cb_scores(user_id)
            rec_df['cb_score'] = cb_scores

            # 3. Context Scores
            rec_df['context_score'] = self._get_context_scores(user_id, rec_df.index)

            # 4. Filter watched movies
            watched_movies = set(
                self.artifacts['ratings'][self.artifacts['ratings']['UserID'] == user_id]['MovieID']
            )
            rec_df = rec_df.drop(watched_movies, errors='ignore')

            if rec_df.empty:
                self.logger.warning(f"No unwatched movies for user {user_id}")
                return None

            # 5. Handle missing values
            rec_df = rec_df.fillna(0.0)

            # 6. Normalize scores
            rec_df = self._normalize_scores(rec_df)

            # 7. Compute final weighted score
            weights = self.config['weights']
            rec_df['final_score'] = (
                weights['cf'] * rec_df['cf_score'] +
                weights['cb'] * rec_df['cb_score'] +
                weights['context'] * rec_df['context_score']
            )

            # 8. Get top recommendations
            top_recommendations = rec_df.nlargest(top_n, 'final_score')

            # 9. Prepare final output
            result = self.artifacts['movies'].loc[top_recommendations.index].copy()
            result['final_score'] = top_recommendations['final_score']
            result['cf_score'] = top_recommendations['cf_score']
            result['cb_score'] = top_recommendations['cb_score']
            result['context_score'] = top_recommendations['context_score']

            self.logger.info(f"Generated {len(result)} recommendations for user {user_id}")

            return result.sort_values('final_score', ascending=False)

        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return None

    def _calculate_ndcg(self, user_ratings: pd.DataFrame, recommendations: pd.DataFrame, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG) for recommendations."""
        try:
            # Create ideal ranking (sorted by actual ratings)
            ideal_ranking = user_ratings.sort_values('Rating', ascending=False)

            # Get actual ranking from recommendations
            actual_ranking = recommendations.index.tolist()

            # Calculate DCG for actual ranking
            dcg = 0
            for i, movie_id in enumerate(actual_ranking[:k]):
                if movie_id in user_ratings['MovieID'].values:
                    rating = user_ratings[user_ratings['MovieID'] == movie_id]['Rating'].iloc[0]
                    dcg += (2 ** rating - 1) / np.log2(i + 2)

            # Calculate IDCG for ideal ranking
            idcg = 0
            for i, (_, row) in enumerate(ideal_ranking.iterrows()):
                if i >= k:
                    break
                idcg += (2 ** row['Rating'] - 1) / np.log2(i + 2)

            # Return NDCG
            return dcg / idcg if idcg > 0 else 0

        except Exception as e:
            self.logger.error(f"Error calculating NDCG: {e}")
            return 0.0

    def _get_cf_scores(self, user_idx: int) -> pd.Series:
        """Get collaborative filtering scores with proper index alignment."""
        try:
            cf_model = self.artifacts['cf_model']

            # Ensure user_idx is within bounds
            if user_idx >= cf_model.user_factors.shape[0]:
                self.logger.warning(f"User index {user_idx} out of bounds, using last valid index")
                user_idx = cf_model.user_factors.shape[0] - 1

            # Get user factors and item factors
            user_factors = cf_model.user_factors[user_idx]
            item_factors = cf_model.item_factors

            # Compute scores
            cf_scores_array = item_factors.dot(user_factors)

            # Create proper index mapping
            movie_ids = [self.artifacts['movie_map_inv'][i] for i in range(len(cf_scores_array))]

            return pd.Series(cf_scores_array, index=movie_ids, name='cf_score')

        except Exception as e:
            self.logger.error(f"Error computing CF scores: {e}")
            return pd.Series(0.0, index=self.artifacts['movies'].index, name='cf_score')

    def _get_cb_scores(self, user_id: int) -> pd.Series:
        """Get content-based scores with error handling."""
        try:
            if user_id in self.artifacts['user_cb_profiles']:
                user_profile = self.artifacts['user_cb_profiles'][user_id]
            else:
                user_profile = self.artifacts['global_cb_profile']

            # Convert to numpy array if needed
            if hasattr(user_profile, 'toarray'):
                user_profile = user_profile.toarray()
            user_profile = np.asarray(user_profile)

            # Convert tfidf_matrix to array if needed
            tfidf_matrix = self.artifacts['tfidf_matrix']
            if hasattr(tfidf_matrix, 'toarray'):
                tfidf_matrix = tfidf_matrix.toarray()
            tfidf_matrix = np.asarray(tfidf_matrix)

            # Compute similarity
            cb_similarities = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()

            # Map back to movie IDs
            movie_ids = [self.artifacts['movie_map_inv'][i] for i in range(len(cb_similarities))]

            return pd.Series(cb_similarities, index=movie_ids, name='cb_score')

        except Exception as e:
            self.logger.error(f"Error computing CB scores: {e}")
            return pd.Series(0.0, index=self.artifacts['movies'].index, name='cb_score')

    def _normalize_scores(self, rec_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize scores with robust handling of edge cases."""

        try:
            scaler = MinMaxScaler()

            # Only normalize if we have variance
            for col in ['cf_score', 'cb_score']:
                if col in rec_df.columns:
                    values = rec_df[col].values.reshape(-1, 1)

                    # Check if all values are the same
                    if np.var(values) > 1e-10:
                        rec_df[col] = scaler.fit_transform(values).flatten()
                    else:
                        # If no variance, set to 0.5
                        rec_df[col] = 0.5

            return rec_df

        except Exception as e:
            self.logger.error(f"Error normalizing scores: {e}")
            return rec_df

    def _cold_start_recommendations(self, user_id: int, top_n: int = 10) -> Optional[pd.DataFrame]:
        """Handle recommendations for new users (cold start problem)."""

        self.logger.info(f"Generating cold start recommendations for user {user_id}")

        try:
            # Get popular movies (high average rating with sufficient ratings)
            ratings = self.artifacts['ratings']
            movie_stats = ratings.groupby('MovieID').agg({
                'Rating': ['mean', 'count']
            }).round(2)

            movie_stats.columns = ['avg_rating', 'rating_count']

            # Filter movies with sufficient ratings
            min_ratings = max(10, len(ratings) // 1000)  # Dynamic threshold
            popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]

            # Sort by average rating
            popular_movies = popular_movies.sort_values('avg_rating', ascending=False)

            # Get top movies
            top_movie_ids = popular_movies.head(top_n).index

            # Prepare result
            result = self.artifacts['movies'].loc[top_movie_ids].copy()
            result['final_score'] = popular_movies.head(top_n)['avg_rating'] / 5.0  # Normalize to 0-1
            result['cf_score'] = 0.0
            result['cb_score'] = 0.0
            result['context_score'] = 0.0

            self.logger.info(f"Generated {len(result)} cold start recommendations")

            return result

        except Exception as e:
            self.logger.error(f"Error generating cold start recommendations: {e}")
            return None

    def _save_artifacts(self):
        """Save artifacts with metadata and error handling."""

        try:
            artifacts_path = self.config['artifacts_path']
            artifacts_path.mkdir(parents=True, exist_ok=True)

            # Save each artifact
            for name, artifact in self.artifacts.items():
                file_path = artifacts_path / f"{name}.joblib"
                joblib.dump(artifact, file_path)

            # Save configuration and metadata
            config_path = artifacts_path / "config.joblib"
            joblib.dump(self.config, config_path)

            self.logger.info(f"Artifacts saved to {artifacts_path}")

        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            raise

    def _load_artifacts(self) -> bool:
        """Load artifacts with comprehensive validation."""

        artifacts_path = self.config['artifacts_path']

        if not artifacts_path.exists():
            self.logger.info(f"Artifacts path {artifacts_path} does not exist")
            return False

        try:
            self.logger.info(f"Loading artifacts from {artifacts_path}")

            # Check for required files
            required_files = [
                'users.joblib', 'movies.joblib', 'ratings.joblib',
                'cf_model.joblib', 'user_cb_profiles.joblib',
                'user_map.joblib', 'movie_map.joblib',
                'tfidf_matrix.joblib'
            ]

            missing_files = [f for f in required_files if not (artifacts_path / f).exists()]
            if missing_files:
                self.logger.warning(f"Missing artifact files: {missing_files}")
                return False

            # Load configuration if available
            config_path = artifacts_path / "config.joblib"
            if config_path.exists():
                saved_config = joblib.load(config_path)
                # Validate compatibility
                if saved_config.get('model_version') != self.config['model_version']:
                    self.logger.warning("Model version mismatch, retraining required")
                    return False

            # Load all artifacts
            for file_path in artifacts_path.glob('*.joblib'):
                if file_path.name != 'config.joblib':
                    try:
                        self.artifacts[file_path.stem] = joblib.load(file_path)
                    except Exception as e:
                        self.logger.error(f"Error loading {file_path.name}: {e}")
                        return False

            # Validate loaded artifacts
            if not self._validate_artifacts():
                return False

            self.logger.info("All artifacts loaded and validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading artifacts: {e}")
            return False

    def _validate_artifacts(self) -> bool:
        """Validate loaded artifacts for consistency."""

        required_artifacts = [
            'users', 'movies', 'ratings', 'cf_model', 'user_cb_profiles',
            'user_map', 'movie_map', 'user_map_inv', 'movie_map_inv', 'tfidf_matrix'
        ]

        for artifact_name in required_artifacts:
            if artifact_name not in self.artifacts:
                self.logger.error(f"Missing required artifact: {artifact_name}")
                return False

        # Validate mappings consistency
        try:
            user_map = self.artifacts['user_map']
            user_map_inv = self.artifacts['user_map_inv']
            movie_map = self.artifacts['movie_map']
            movie_map_inv = self.artifacts['movie_map_inv']

            # Check inverse mappings
            if len(user_map) != len(user_map_inv):
                self.logger.error("User mapping inconsistency")
                return False

            if len(movie_map) != len(movie_map_inv):
                self.logger.error("Movie mapping inconsistency")
                return False

            # Check model dimensions
            cf_model = self.artifacts['cf_model']
            expected_users = len(user_map)
            expected_movies = len(movie_map)

            if cf_model.user_factors.shape[0] != expected_users:
                self.logger.error("CF model user dimension mismatch")
                return False

            if cf_model.item_factors.shape[0] != expected_movies:
                self.logger.error("CF model item dimension mismatch")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Artifact validation error: {e}")
            return False

    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information and statistics."""

        if user_id not in self.artifacts['users'].index:
            return None

        user_data = self.artifacts['users'].loc[user_id]
        user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['UserID'] == user_id]

        # Age mapping for display
        age_map = {
            1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
            45: "45-49", 50: "50-55", 56: "56+"
        }

        return {
            'user_id': user_id,
            'gender': user_data['Gender'],
            'age': user_data['Age'],
            'age_group': age_map.get(user_data['Age'], 'Unknown'),
            'occupation': user_data['Occupation'],
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['Rating'].mean() if not user_ratings.empty else 0,
            'rating_distribution': user_ratings['Rating'].value_counts().to_dict() if not user_ratings.empty else {}
        }

    def get_movie_info(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Get movie information and statistics."""

        if movie_id not in self.artifacts['movies'].index:
            return None

        movie_data = self.artifacts['movies'].loc[movie_id]
        movie_ratings = self.artifacts['ratings'][self.artifacts['ratings']['MovieID'] == movie_id]

        return {
            'movie_id': movie_id,
            'title': movie_data['Title'],
            'genres': movie_data['Genres'].split('|'),
            'total_ratings': len(movie_ratings),
            'avg_rating': movie_ratings['Rating'].mean() if not movie_ratings.empty else 0,
            'rating_distribution': movie_ratings['Rating'].value_counts().to_dict() if not movie_ratings.empty else {}
        }

    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""

        if not self.is_trained:
            return {'status': 'not_trained'}

        ratings = self.artifacts['ratings']
        users = self.artifacts['users']
        movies = self.artifacts['movies']

        # Calculate statistics
        stats = {
            'model_version': self.config['model_version'],
            'created_at': self.config['created_at'],
            'total_users': len(users),
            'total_movies': len(movies),
            'total_ratings': len(ratings),
            'rating_density': len(ratings) / (len(users) * len(movies)),
            'avg_ratings_per_user': len(ratings) / len(users),
            'avg_ratings_per_movie': len(ratings) / len(movies),
            'rating_distribution': ratings['Rating'].value_counts().to_dict(),
            'config': self.config.copy()
        }

        # Add genre statistics
        all_genres = []
        for genres_str in movies['Genres']:
            all_genres.extend(genres_str.split('|'))

        genre_counts = pd.Series(all_genres).value_counts()
        stats['top_genres'] = genre_counts.head(10).to_dict()

        return stats

    def evaluate_system_accuracy(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Evaluate the entire recommendation system with comprehensive metrics."""

        if not self.is_trained:
            self.logger.error("Model must be trained before evaluation")
            return {}

        try:
            self.logger.info("Starting comprehensive system evaluation...")

            # Split ratings for evaluation
            ratings = self.artifacts['ratings'].copy()
            train_ratings, test_ratings = train_test_split(
                ratings, test_size=test_size, random_state=random_state, stratify=ratings['UserID']
            )

            # Store original ratings and temporarily use train set
            original_ratings = self.artifacts['ratings']
            self.artifacts['ratings'] = train_ratings

            # Initialize metrics
            metrics = {
                'precision': [],
                'recall': [],
                'ndcg': [],
                'coverage': set(),
                'diversity': [],
                'novelty': [],
                'serendipity': [],
                'personalization': [],
                'cold_start_performance': {
                    'precision': [],
                    'recall': [],
                    'ndcg': []
                }
            }

            # Get valid user IDs that exist in both user_map and test_ratings
            valid_user_ids = set(self.artifacts['user_map'].keys()).intersection(
                set(test_ratings['UserID'].unique())
            )

            # Sample users for evaluation (limit to 100)
            test_users = list(valid_user_ids)[:100]

            if not test_users:
                self.logger.error("No valid users found for evaluation")
                return {}

            # Calculate global movie popularity for novelty
            movie_popularity = ratings['MovieID'].value_counts(normalize=True)

            # Track all recommendations for personalization
            all_recommendations = []

            for user_id in test_users:
                try:
                    user_test_ratings = test_ratings[test_ratings['UserID'] == user_id]
                    liked_movies = set(user_test_ratings[user_test_ratings['Rating'] >= 4]['MovieID'])

                    if len(liked_movies) == 0:
                        continue

                    # Get recommendations
                    recs = self.get_recommendations(user_id, top_n=10)
                    if recs is None or recs.empty:
                        continue

                    # Ensure all recommended movies exist in the movies DataFrame
                    valid_recs = recs[recs.index.isin(self.artifacts['movies'].index)]
                    if valid_recs.empty:
                        continue

                    recommended_movies = set(valid_recs.index)
                    all_recommendations.append(recommended_movies)

                    # Basic metrics
                    hits = len(liked_movies.intersection(recommended_movies))
                    precision = hits / len(recommended_movies) if recommended_movies else 0
                    recall = hits / len(liked_movies) if liked_movies else 0

                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)

                    # NDCG
                    ndcg = self._calculate_ndcg(user_test_ratings, valid_recs)
                    metrics['ndcg'].append(ndcg)

                    # Coverage
                    metrics['coverage'].update(recommended_movies)

                    # Diversity (genre diversity)
                    genres = valid_recs['Genres'].str.split('|')
                    unique_genres = set()
                    for genre_list in genres:
                        unique_genres.update(genre_list)
                    metrics['diversity'].append(len(unique_genres) / len(valid_recs))

                    # Novelty (inverse popularity)
                    valid_popularity = movie_popularity[movie_popularity.index.isin(valid_recs.index)]
                    if not valid_popularity.empty:
                        novelty = 1 - valid_popularity.mean()
                        metrics['novelty'].append(novelty)

                    # Serendipity (unexpected but relevant recommendations)
                    user_rated_movies = set(train_ratings[train_ratings['UserID'] == user_id]['MovieID'])
                    unexpected_movies = recommended_movies - user_rated_movies
                    if unexpected_movies:
                        unexpected_ratings = user_test_ratings[user_test_ratings['MovieID'].isin(unexpected_movies)]
                        serendipity = len(unexpected_ratings[unexpected_ratings['Rating'] >= 4]) / len(unexpected_movies)
                        metrics['serendipity'].append(serendipity)

                    # Cold start evaluation (users with few ratings)
                    if len(user_rated_movies) <= 5:
                        metrics['cold_start_performance']['precision'].append(precision)
                        metrics['cold_start_performance']['recall'].append(recall)
                        metrics['cold_start_performance']['ndcg'].append(ndcg)

                except Exception as e:
                    self.logger.error(f"Error processing user {user_id}: {e}")
                    continue

            # Calculate personalization (diversity between users' recommendations)
            personalization = 0
            if len(all_recommendations) > 1:
                for i in range(len(all_recommendations)):
                    for j in range(i + 1, len(all_recommendations)):
                        intersection = len(all_recommendations[i].intersection(all_recommendations[j]))
                        union = len(all_recommendations[i].union(all_recommendations[j]))
                        if union > 0:
                            personalization += 1 - (intersection / union)
                metrics['personalization'] = personalization / (len(all_recommendations) * (len(all_recommendations) - 1) / 2)
            else:
                metrics['personalization'] = 0

            # Restore original ratings
            self.artifacts['ratings'] = original_ratings

            # Calculate final metrics
            results = {
                'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
                'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
                'ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0,
                'coverage': len(metrics['coverage']) / len(self.artifacts['movies']),
                'diversity': np.mean(metrics['diversity']) if metrics['diversity'] else 0,
                'novelty': np.mean(metrics['novelty']) if metrics['novelty'] else 0,
                'serendipity': np.mean(metrics['serendipity']) if metrics['serendipity'] else 0,
                'personalization': metrics['personalization'],
                'cold_start_performance': {
                    'precision': np.mean(metrics['cold_start_performance']['precision']) if metrics['cold_start_performance']['precision'] else 0,
                    'recall': np.mean(metrics['cold_start_performance']['recall']) if metrics['cold_start_performance']['recall'] else 0,
                    'ndcg': np.mean(metrics['cold_start_performance']['ndcg']) if metrics['cold_start_performance']['ndcg'] else 0
                },
                'evaluated_users': len(metrics['precision'])
            }

            # Calculate F1 score
            results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0

            self.logger.info("Comprehensive evaluation completed")
            return results

        except Exception as e:
            self.logger.error(f"Error during comprehensive evaluation: {e}")
            return {}

    def print_accuracy_metrics(self):
        """Print comprehensive accuracy metrics in a formatted way."""

        print("\n" + "=" * 80)
        print("RECOMMENDATION SYSTEM ACCURACY METRICS")
        print("=" * 80)

        metrics = self.evaluate_system_accuracy()

        if not metrics:
            print("Error: Could not calculate metrics")
            return

        # Basic Metrics
        print("\nBasic Metrics:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"NDCG: {metrics['ndcg']:.3f}")

        # Coverage and Diversity
        print("\nCoverage and Diversity:")
        print(f"Catalog Coverage: {metrics['coverage']:.1%}")
        print(f"Genre Diversity: {metrics['diversity']:.3f}")
        print(f"Novelty: {metrics['novelty']:.3f}")

        # Advanced Metrics
        print("\nAdvanced Metrics:")
        print(f"Serendipity: {metrics['serendipity']:.3f}")
        print(f"Personalization: {metrics['personalization']:.3f}")

        # Cold Start Performance
        print("\nCold Start Performance:")
        cold_start = metrics['cold_start_performance']
        print(f"Precision: {cold_start['precision']:.3f}")
        print(f"Recall: {cold_start['recall']:.3f}")
        print(f"NDCG: {cold_start['ndcg']:.3f}")

        # Evaluation Details
        print("\nEvaluation Details:")
        print(f"Evaluated Users: {metrics['evaluated_users']}")

        print("\n" + "=" * 80)

    def get_movie_based_recommendations(self, movie_id_or_name: Union[int, str], top_n: int = 10) -> Optional[pd.DataFrame]:
        """Get recommendations based on a selected movie."""

        try:
            # Find movie ID if name is provided
            if isinstance(movie_id_or_name, str):
                # Try exact match first
                matches = self.artifacts['movies'][self.artifacts['movies']['Title'].str.lower() == movie_id_or_name.lower()]
                if matches.empty:
                    # Try partial match
                    matches = self.artifacts['movies'][self.artifacts['movies']['Title'].str.lower().str.contains(movie_id_or_name.lower())]
                if matches.empty:
                    self.logger.warning(f"No movie found matching: {movie_id_or_name}")
                    return None
                movie_id = matches.index[0]
            else:
                movie_id = movie_id_or_name

            if movie_id not in self.artifacts['movies'].index:
                self.logger.warning(f"Movie ID {movie_id} not found")
                return None

            # Get movie's TF-IDF vector
            movie_idx = self.artifacts['movies'].index.get_loc(movie_id)
            movie_vec = self.artifacts['tfidf_matrix'][movie_idx]

            # Calculate similarity with all movies
            similarities = cosine_similarity(movie_vec, self.artifacts['tfidf_matrix']).flatten()

            # Get top N similar movies (excluding the input movie)
            similar_indices = similarities.argsort()[::-1][1:top_n+1]
            similar_movie_ids = self.artifacts['movies'].index[similar_indices]

            # Prepare result
            result = self.artifacts['movies'].loc[similar_movie_ids].copy()
            result['similarity_score'] = similarities[similar_indices]

            # Add additional information
            result['avg_rating'] = result.index.map(
                lambda x: self.artifacts['ratings'][self.artifacts['ratings']['MovieID'] == x]['Rating'].mean()
            )
            result['rating_count'] = result.index.map(
                lambda x: len(self.artifacts['ratings'][self.artifacts['ratings']['MovieID'] == x])
            )

            return result.sort_values('similarity_score', ascending=False)

        except Exception as e:
            self.logger.error(f"Error getting movie-based recommendations: {e}")
            return None

    def get_homepage_sections(self, top_n=10):
        """Return homepage sections for movies, books, and songs, including new releases and top by genre."""
        sections = {}
        try:
            # --- Popular Movies ---
            if hasattr(self, 'ratings') and not self.ratings.empty:
                movie_stats = self.ratings.groupby('MovieID').agg({'Rating': ['mean', 'count']})
                movie_stats.columns = ['avg_rating', 'rating_count']
                popular_movies = movie_stats.sort_values(['avg_rating', 'rating_count'], ascending=[False, False]).head(top_n)
                movie_info = self.artifacts['movies'].loc[popular_movies.index]
                sections['Popular Movies'] = movie_info[['Title', 'Genres']].assign(avg_rating=popular_movies['avg_rating'].values)

            # --- Popular Books ---
            if hasattr(self, 'books') and 'Rating' in self.books.columns:
                book_stats = self.books.groupby('Id').agg({'Rating': ['mean', 'count']})
                book_stats.columns = ['avg_rating', 'rating_count']
                popular_books = book_stats.sort_values(['avg_rating', 'rating_count'], ascending=[False, False]).head(top_n)
                book_info = self.books.set_index('Id').loc[popular_books.index]
                sections['Popular Books'] = book_info[['Name', 'Authors', 'Genre']].assign(avg_rating=popular_books['avg_rating'].values)

            # --- Popular Songs ---
            if hasattr(self, 'listening_history') and 'Rating' in self.listening_history.columns:
                song_stats = self.listening_history.groupby('track_id').agg({'Rating': ['mean', 'count']})
                song_stats.columns = ['avg_rating', 'rating_count']
                popular_songs = song_stats.sort_values(['avg_rating', 'rating_count'], ascending=[False, False]).head(top_n)
                song_info = self.music_info.set_index('track_id').loc[popular_songs.index]
                sections['Popular Songs'] = song_info[['name', 'artist', 'genre']].assign(avg_rating=popular_songs['avg_rating'].values)

            # --- New Releases: Movies ---
            if hasattr(self, 'artifacts') and 'movies' in self.artifacts:
                movies = self.artifacts['movies'].copy()
                movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
                new_movies = movies.sort_values('Year', ascending=False).head(top_n)
                sections['New Movie Releases'] = new_movies[['Title', 'Genres', 'Year']]

            # --- New Releases: Books ---
            if hasattr(self, 'books'):
                books = self.books.copy()
                if 'Year' in books.columns:
                    new_books = books.sort_values('Year', ascending=False).head(top_n)
                else:
                    new_books = books.sort_values('Id', ascending=False).head(top_n)
                sections['New Book Releases'] = new_books[['Name', 'Authors', 'Genre']]

            # --- New Releases: Songs ---
            if hasattr(self, 'music_info'):
                songs = self.music_info.copy()
                if 'year' in songs.columns:
                    new_songs = songs.sort_values('year', ascending=False).head(top_n)
                else:
                    new_songs = songs.sort_values('track_id', ascending=False).head(top_n)
                sections['New Song Releases'] = new_songs[['name', 'artist', 'genre']]

            # --- Top by Genre: Movies ---
            if hasattr(self, 'artifacts') and 'movies' in self.artifacts:
                genre_counts = self.artifacts['movies']['Genres'].str.split('|').explode().value_counts()
                top_genres = genre_counts.head(2).index.tolist()
                for genre in top_genres:
                    genre_movies = self.artifacts['movies'][self.artifacts['movies']['Genres'].str.contains(genre, na=False)]
                    genre_movies = genre_movies.copy()
                    genre_movies['avg_rating'] = genre_movies.index.map(
                        self.artifacts['ratings'].groupby('MovieID')['Rating'].mean()
                    )
                    top_genre_movies = genre_movies.sort_values('avg_rating', ascending=False).head(top_n)
                    sections[f'Top {genre} Movies'] = top_genre_movies[['Title', 'Genres', 'avg_rating']]

            # --- Top by Genre: Books ---
            if hasattr(self, 'books'):
                genre_counts = self.books['Genre'].str.split(',').explode().value_counts()
                top_genres = genre_counts.head(2).index.tolist()
                for genre in top_genres:
                    genre_books = self.books[self.books['Genre'].str.contains(genre, na=False)]
                    genre_books = genre_books.copy()
                    if 'Rating' in genre_books.columns:
                        genre_books['avg_rating'] = genre_books.groupby('Id')['Rating'].transform('mean')
                    top_genre_books = genre_books.sort_values('avg_rating', ascending=False).head(top_n)
                    sections[f'Top {genre} Books'] = top_genre_books[['Name', 'Authors', 'Genre', 'avg_rating']]

            # --- Top by Genre: Songs ---
            if hasattr(self, 'music_info'):
                genre_counts = self.music_info['genre'].str.split(',').explode().value_counts()
                top_genres = genre_counts.head(2).index.tolist()
                for genre in top_genres:
                    genre_songs = self.music_info[self.music_info['genre'].str.contains(genre, na=False)]
                    genre_songs = genre_songs.copy()
                    if 'track_id' in genre_songs.columns and hasattr(self, 'listening_history'):
                        genre_songs['avg_rating'] = genre_songs['track_id'].map(
                            self.listening_history.groupby('track_id')['Rating'].mean()
                        )
                    top_genre_songs = genre_songs.sort_values('avg_rating', ascending=False).head(top_n)
                    sections[f'Top {genre} Songs'] = top_genre_songs[['name', 'artist', 'genre', 'avg_rating']]

        except Exception as e:
            self.logger.error(f"Error generating homepage sections: {e}")

        return sections

    def load_cross_domain_data(self):
        """Load data for books and songs domains and fit TF-IDF ONCE for cross-domain recommendations."""
        try:
            # Load book and music data if not already loaded
            if not hasattr(self, 'books') or self.books is None:
                self.books = pd.read_csv('/content/book1-100k.csv')
            if not hasattr(self, 'music_info') or self.music_info is None:
                self.music_info = pd.read_csv('/content/Music Info.csv')
            if not hasattr(self, 'listening_history') or self.listening_history is None:
                self.listening_history = pd.read_csv('/content/User Listening History.csv')

            # Create movies_items from artifacts
            movies_items = self.artifacts['movies'].reset_index()
            movies_items['domain'] = 'movies'
            movies_items['id'] = movies_items['MovieID'].astype(str)
            movies_items['name'] = movies_items['Title']
            movies_items['features'] = movies_items['Genres'].str.replace('|', ' ')

            # Create books_items from self.books
            books_items = self.books.copy()
            books_items['domain'] = 'books'
            books_items['id'] = books_items['Id'].astype(str)
            books_items['name'] = books_items['Name']
            def safe_col(df, col):
                return df[col].fillna('') if col in df else pd.Series([''] * len(df), index=df.index)
            books_items['features'] = (
                safe_col(books_items, 'Genre') + ' ' +
                safe_col(books_items, 'Description') + ' ' +
                safe_col(books_items, 'Authors')
            ).str.strip()

            # Create music_items from self.music_info
            music_items = self.music_info.copy()
            music_items['domain'] = 'songs'
            music_items['id'] = music_items['track_id'].astype(str)
            music_items['name'] = music_items['name']
            music_items['features'] = (
                music_items['artist'].fillna('') + ' ' +
                music_items['genre'].fillna('') + ' ' +
                music_items.get('tags', pd.Series(['']*len(music_items))).fillna('')
            ).str.strip()

            # Combine all items with error handling
            self.all_items = pd.concat([
                movies_items[['id', 'name', 'features', 'domain']],
                books_items[['id', 'name', 'features', 'domain']],
                music_items[['id', 'name', 'features', 'domain']]
            ], ignore_index=True)
            # Fit TF-IDF ONCE and store
            self.cross_tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
            self.cross_features_matrix = self.cross_tfidf.fit_transform(self.all_items['features'].fillna(''))
            self.logger.info("Cross-domain TF-IDF fitted and stored.")
        except Exception as e:
            self.logger.error(f"Error loading cross-domain data: {str(e)}")
            raise

    def get_cross_domain_recommendations(self, item_id: str, domain: str, n_recommendations: int = 10) -> pd.DataFrame:
        try:
            if not hasattr(self, 'all_items') or not hasattr(self, 'cross_features_matrix'):
                self.logger.error("Cross-domain features not available")
                return pd.DataFrame()
            items = self.all_items
            item_id = str(item_id)
            source_item = items[(items['id'] == item_id) & (items['domain'] == domain)]
            if source_item.empty:
                self.logger.error(f"Item {item_id} not found in {domain} domain")
                print(f"DEBUG: Item id '{item_id}' not found in all_items for domain '{domain}'. Available ids: {items[items['domain'] == domain]['id'].head(10).tolist()}")
                return pd.DataFrame()
            item_idx = source_item.index[0]
            item_features = self.cross_features_matrix[item_idx]
            similarities = cosine_similarity(item_features, self.cross_features_matrix).flatten()
            similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
            recommendations = items.iloc[similar_indices].copy()
            recommendations['similarity_score'] = similarities[similar_indices]
            # Optionally add avg_rating/rating_count if available
            return recommendations
        except Exception as e:
            self.logger.error(f"Error getting cross-domain recommendations: {e}")
            print(f"DEBUG: Exception in get_cross_domain_recommendations: {e}")
            return pd.DataFrame()

    def get_homepage_sections(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        try:
            sections = {}
            ratings = self.artifacts['ratings']
            movie_stats = ratings.groupby('MovieID').agg({
                'Rating': ['mean', 'count']
            }).round(2)
            movie_stats.columns = ['avg_rating', 'rating_count']
            min_ratings = max(10, len(ratings) // 1000)
            popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
            popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
            valid_movie_ids = popular_movies.index.intersection(self.artifacts['movies'].index)
            if len(valid_movie_ids) > 0:
                sections['Popular Movies'] = self.artifacts['movies'].loc[valid_movie_ids[:top_n]].copy()
                sections['Popular Movies']['avg_rating'] = popular_movies.loc[valid_movie_ids[:top_n], 'avg_rating']
            all_genres = set()
            for genres in self.artifacts['movies']['Genres']:
                all_genres.update(genres.split('|'))
            for genre in ['Action', 'Comedy', 'Drama', 'Sci-Fi']:
                if genre in all_genres:
                    genre_movies = self.artifacts['movies'][
                        self.artifacts['movies']['Genres'].str.contains(genre)
                    ]
                    valid_genre_ids = genre_movies.index.intersection(movie_stats.index)
                    if len(valid_genre_ids) > 0:
                        genre_ratings = movie_stats.loc[valid_genre_ids]
                        top_genre_movies = genre_ratings.sort_values('avg_rating', ascending=False)
                        valid_top_ids = top_genre_movies.index.intersection(self.artifacts['movies'].index)
                        if len(valid_top_ids) > 0:
                            sections[f'Top {genre} Movies'] = self.artifacts['movies'].loc[valid_top_ids[:top_n]].copy()
                            sections[f'Top {genre} Movies']['avg_rating'] = top_genre_movies.loc[valid_top_ids[:top_n], 'avg_rating']
            return sections
        except Exception as e:
            self.logger.error(f"Error generating homepage sections: {e}")
            return {}

    def get_structured_recommendations(self, item_id, domain, n_main=10, n_cross=5):
        """
        For a given item and domain, return structured recommendations:
        - n_main from the same domain
        - n_cross from each other domain
        Each recommendation includes: title/name, genres, avg rating, rating count, similarity score, etc.
        Returns a dict: {'movies': [...], 'books': [...], 'songs': [...]}.
        """
        result = {'movies': [], 'books': [], 'songs': []}
        domain_list = ['movies', 'books', 'songs']
        # Helper to extract details from a row
        def extract_details(row, domain, sim_score=None):
            details = {}
            if domain == 'movies':
                details['title'] = row.get('Title') or row.get('title')
                details['genres'] = row.get('Genres') or row.get('genre')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            elif domain == 'books':
                details['title'] = row.get('Name') or row.get('Title') or row.get('title')
                details['genres'] = row.get('Genre') or row.get('Genres') or row.get('genre')
                details['authors'] = row.get('Authors') or row.get('authors')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            elif domain == 'songs':
                details['title'] = row.get('name') or row.get('Title') or row.get('title')
                details['genres'] = row.get('genre') or row.get('Genres')
                details['artist'] = row.get('artist')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            if sim_score is not None:
                details['similarity_score'] = sim_score
            return details
        # Main domain recommendations
        if domain == 'movies':
            main_recs = self.get_movie_based_recommendations(item_id, top_n=n_main)
            if main_recs is not None and not main_recs.empty:
                for _, row in main_recs.iterrows():
                    result['movies'].append(extract_details(row, 'movies', row.get('similarity_score')))
        else:
            main_recs = self.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=n_main)
            if main_recs is not None and not main_recs.empty:
                for _, row in main_recs.iterrows():
                    d = row.get('domain', domain)
                    if d in result:
                        result[d].append(extract_details(row, d, row.get('similarity_score')))
        # Cross-domain recommendations
        for other_domain in domain_list:
            if other_domain == domain:
                continue
            cross_recs = self.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=n_cross)
            if cross_recs is not None and not cross_recs.empty:
                for _, row in cross_recs.iterrows():
                    d = row.get('domain', other_domain)
                    if d in result:
                        result[d].append(extract_details(row, d, row.get('similarity_score')))
        return result

    def _get_context_scores(self, user_id: int, candidate_movie_ids: pd.Index) -> pd.Series:
        """Data-driven context-aware scores based on user's actual genre preferences."""
        try:
            user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['UserID'] == user_id]
            if user_ratings.empty:
                return pd.Series(0.0, index=candidate_movie_ids, name='context_score')
            # Compute user's average rating per genre
            movie_genres = self.artifacts['movies']['Genres']
            genre_ratings = {}
            for _, row in user_ratings.iterrows():
                movie_id = row['MovieID']
                rating = row['Rating']
                genres = movie_genres.get(movie_id, '')
                for genre in genres.split('|'):
                    if genre:
                        genre_ratings.setdefault(genre, []).append(rating)
            # Average rating per genre
            genre_avg = {g: np.mean(r) for g, r in genre_ratings.items() if r}
            # Score each candidate movie by averaging user's avg rating for its genres
            scores = {}
            for movie_id in candidate_movie_ids:
                genres = movie_genres.get(movie_id, '')
                genre_list = [g for g in genres.split('|') if g]
                if genre_list:
                    genre_scores = [genre_avg.get(g, np.mean(list(genre_avg.values())) if genre_avg else 0.0) for g in genre_list]
                    scores[movie_id] = np.mean(genre_scores)
                else:
                    scores[movie_id] = np.mean(list(genre_avg.values())) if genre_avg else 0.0
            # Normalize scores to 0-1
            vals = np.array(list(scores.values()))
            if len(vals) > 1 and np.ptp(vals) > 0:
                minv, maxv = np.min(vals), np.max(vals)
                for k in scores:
                    scores[k] = (scores[k] - minv) / (maxv - minv)
            return pd.Series(scores, name='context_score').reindex(candidate_movie_ids).fillna(0.0)
        except Exception as e:
            self.logger.warning(f"Error computing data-driven context scores: {e}")
            return pd.Series(0.0, index=candidate_movie_ids, name='context_score')

class UnifiedRecommender(HybridRecommenderV3):
    """
    A unified recommendation system that can recommend items across all domains
    (movies, books, songs) based on the current item.
    """
    DOMAIN_CONFIGS = {
        'movies': {
            'required_columns': ['MovieID', 'Title', 'Genres'],
            'id_col': 'MovieID',
            'title_col': 'Title',
            'genre_col': 'Genres',
            'rating_col': 'Rating',
            'user_col': 'UserID'
        },
        'books': {
            'required_columns': ['Id', 'Name', 'Authors', 'Genre', 'Description'],
            'id_col': 'Id',
            'title_col': 'Name',
            'genre_col': 'Genre',
            'rating_col': 'Rating',
            'user_col': 'UserID'
        },
        'songs': {
            'required_columns': ['track_id', 'name', 'artist', 'genre'],
            'id_col': 'track_id',
            'title_col': 'name',
            'genre_col': 'genre',
            'rating_col': 'Rating',
            'user_col': 'user_id'
        }
    }

    def __init__(self, data_paths: dict, artifacts_path: str = './recommender_artifacts_unified', als_factors: int = 100, als_iterations: int = 25, als_regularization: float = 0.01, weights: dict = None, enable_logging: bool = True):
        self.all_data_paths = data_paths
        self.domain_data = {}
        self.unified_data = None
        movie_paths = data_paths.get('movies', {})
        if not movie_paths:
            raise ValueError("Movie data paths are required")
        super().__init__(
            data_paths=movie_paths,
            artifacts_path=artifacts_path,
            als_factors=als_factors,
            als_iterations=als_iterations,
            als_regularization=als_regularization,
            weights=weights,
            enable_logging=enable_logging
        )
        self.load_cross_domain_data()

    def _load_domain_data(self, domain: str, paths: dict):
        try:
            config = self.DOMAIN_CONFIGS.get(domain)
            if not config:
                self.logger.error(f"No configuration found for domain: {domain}")
                return None
            domain_data = {}
            if 'items' in paths and os.path.exists(paths['items']):
                try:
                    items = pd.read_csv(paths['items'], encoding='latin-1', on_bad_lines='skip')
                    required_cols = config['required_columns']
                    missing_cols = [col for col in required_cols if col not in items.columns]
                    for col in missing_cols:
                        items[col] = 'Unknown'
                    domain_data['items'] = items
                except Exception as e:
                    self.logger.error(f"Error reading items file for {domain}: {e}")
                    return None
            if 'ratings' in paths and os.path.exists(paths['ratings']):
                try:
                    ratings = pd.read_csv(paths['ratings'], encoding='latin-1', on_bad_lines='skip')
                    domain_data['ratings'] = ratings
                except Exception as e:
                    self.logger.error(f"Error reading ratings file for {domain}: {e}")
                    return None
            if 'users' in paths and os.path.exists(paths['users']):
                try:
                    users = pd.read_csv(paths['users'], encoding='latin-1', on_bad_lines='skip')
                    domain_data['users'] = users
                except Exception as e:
                    self.logger.error(f"Error reading users file for {domain}: {e}")
                    return None
            return domain_data
        except Exception as e:
            self.logger.error(f"Error loading {domain} data: {e}")
            return None

    def load_cross_domain_data(self):
        """Load data for books and songs domains and fit TF-IDF ONCE for cross-domain recommendations."""
        try:
            # Load book and music data if not already loaded
            if not hasattr(self, 'books') or self.books is None:
                self.books = pd.read_csv('/content/book1-100k.csv')
            if not hasattr(self, 'music_info') or self.music_info is None:
                self.music_info = pd.read_csv('/content/Music Info.csv')
            if not hasattr(self, 'listening_history') or self.listening_history is None:
                self.listening_history = pd.read_csv('/content/User Listening History.csv')

            # Create movies_items from artifacts
            movies_items = self.artifacts['movies'].reset_index()
            movies_items['domain'] = 'movies'
            movies_items['id'] = movies_items['MovieID'].astype(str)
            movies_items['name'] = movies_items['Title']
            movies_items['features'] = movies_items['Genres'].str.replace('|', ' ')

            # Create books_items from self.books
            books_items = self.books.copy()
            books_items['domain'] = 'books'
            books_items['id'] = books_items['Id'].astype(str)
            books_items['name'] = books_items['Name']
            def safe_col(df, col):
                return df[col].fillna('') if col in df else pd.Series([''] * len(df), index=df.index)
            books_items['features'] = (
                safe_col(books_items, 'Genre') + ' ' +
                safe_col(books_items, 'Description') + ' ' +
                safe_col(books_items, 'Authors')
            ).str.strip()

            # Create music_items from self.music_info
            music_items = self.music_info.copy()
            music_items['domain'] = 'songs'
            music_items['id'] = music_items['track_id'].astype(str)
            music_items['name'] = music_items['name']
            music_items['features'] = (
                music_items['artist'].fillna('') + ' ' +
                music_items['genre'].fillna('') + ' ' +
                music_items.get('tags', pd.Series(['']*len(music_items))).fillna('')
            ).str.strip()

            # Combine all items with error handling
            self.all_items = pd.concat([
                movies_items[['id', 'name', 'features', 'domain']],
                books_items[['id', 'name', 'features', 'domain']],
                music_items[['id', 'name', 'features', 'domain']]
            ], ignore_index=True)
            # Fit TF-IDF ONCE and store
            self.cross_tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
            self.cross_features_matrix = self.cross_tfidf.fit_transform(self.all_items['features'].fillna(''))
            self.logger.info("Cross-domain TF-IDF fitted and stored.")
        except Exception as e:
            self.logger.error(f"Error loading cross-domain data: {str(e)}")
            raise

    def get_cross_domain_recommendations(self, item_id: str, domain: str, n_recommendations: int = 10) -> pd.DataFrame:
        try:
            if not hasattr(self, 'all_items') or not hasattr(self, 'cross_features_matrix'):
                self.logger.error("Cross-domain features not available")
                return pd.DataFrame()
            items = self.all_items
            item_id = str(item_id)
            source_item = items[(items['id'] == item_id) & (items['domain'] == domain)]
            if source_item.empty:
                self.logger.error(f"Item {item_id} not found in {domain} domain")
                print(f"DEBUG: Item id '{item_id}' not found in all_items for domain '{domain}'. Available ids: {items[items['domain'] == domain]['id'].head(10).tolist()}")
                return pd.DataFrame()
            item_idx = source_item.index[0]
            item_features = self.cross_features_matrix[item_idx]
            similarities = cosine_similarity(item_features, self.cross_features_matrix).flatten()
            similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
            recommendations = items.iloc[similar_indices].copy()
            recommendations['similarity_score'] = similarities[similar_indices]
            # Optionally add avg_rating/rating_count if available
            return recommendations
        except Exception as e:
            self.logger.error(f"Error getting cross-domain recommendations: {e}")
            print(f"DEBUG: Exception in get_cross_domain_recommendations: {e}")
            return pd.DataFrame()

    def get_homepage_sections(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        try:
            sections = {}
            ratings = self.artifacts['ratings']
            movie_stats = ratings.groupby('MovieID').agg({
                'Rating': ['mean', 'count']
            }).round(2)
            movie_stats.columns = ['avg_rating', 'rating_count']
            min_ratings = max(10, len(ratings) // 1000)
            popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
            popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
            valid_movie_ids = popular_movies.index.intersection(self.artifacts['movies'].index)
            if len(valid_movie_ids) > 0:
                sections['Popular Movies'] = self.artifacts['movies'].loc[valid_movie_ids[:top_n]].copy()
                sections['Popular Movies']['avg_rating'] = popular_movies.loc[valid_movie_ids[:top_n], 'avg_rating']
            all_genres = set()
            for genres in self.artifacts['movies']['Genres']:
                all_genres.update(genres.split('|'))
            for genre in ['Action', 'Comedy', 'Drama', 'Sci-Fi']:
                if genre in all_genres:
                    genre_movies = self.artifacts['movies'][
                        self.artifacts['movies']['Genres'].str.contains(genre)
                    ]
                    valid_genre_ids = genre_movies.index.intersection(movie_stats.index)
                    if len(valid_genre_ids) > 0:
                        genre_ratings = movie_stats.loc[valid_genre_ids]
                        top_genre_movies = genre_ratings.sort_values('avg_rating', ascending=False)
                        valid_top_ids = top_genre_movies.index.intersection(self.artifacts['movies'].index)
                        if len(valid_top_ids) > 0:
                            sections[f'Top {genre} Movies'] = self.artifacts['movies'].loc[valid_top_ids[:top_n]].copy()
                            sections[f'Top {genre} Movies']['avg_rating'] = top_genre_movies.loc[valid_top_ids[:top_n], 'avg_rating']
            return sections
        except Exception as e:
            self.logger.error(f"Error generating homepage sections: {e}")
            return {}

    def get_structured_recommendations(self, item_id, domain, n_main=10, n_cross=5):
        """
        For a given item and domain, return structured recommendations:
        - n_main from the same domain
        - n_cross from each other domain
        Each recommendation includes: title/name, genres, avg rating, rating count, similarity score, etc.
        Returns a dict: {'movies': [...], 'books': [...], 'songs': [...]}.
        """
        result = {'movies': [], 'books': [], 'songs': []}
        domain_list = ['movies', 'books', 'songs']
        # Helper to extract details from a row
        def extract_details(row, domain, sim_score=None):
            details = {}
            if domain == 'movies':
                details['title'] = row.get('Title') or row.get('title')
                details['genres'] = row.get('Genres') or row.get('genre')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            elif domain == 'books':
                details['title'] = row.get('Name') or row.get('Title') or row.get('title')
                details['genres'] = row.get('Genre') or row.get('Genres') or row.get('genre')
                details['authors'] = row.get('Authors') or row.get('authors')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            elif domain == 'songs':
                details['title'] = row.get('name') or row.get('Title') or row.get('title')
                details['genres'] = row.get('genre') or row.get('Genres')
                details['artist'] = row.get('artist')
                details['avg_rating'] = row.get('avg_rating')
                details['rating_count'] = row.get('rating_count')
            if sim_score is not None:
                details['similarity_score'] = sim_score
            return details
        # Main domain recommendations
        if domain == 'movies':
            main_recs = self.get_movie_based_recommendations(item_id, top_n=n_main)
            if main_recs is not None and not main_recs.empty:
                for _, row in main_recs.iterrows():
                    result['movies'].append(extract_details(row, 'movies', row.get('similarity_score')))
        else:
            main_recs = self.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=n_main)
            if main_recs is not None and not main_recs.empty:
                for _, row in main_recs.iterrows():
                    d = row.get('domain', domain)
                    if d in result:
                        result[d].append(extract_details(row, d, row.get('similarity_score')))
        # Cross-domain recommendations
        for other_domain in domain_list:
            if other_domain == domain:
                continue
            cross_recs = self.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=n_cross)
            if cross_recs is not None and not cross_recs.empty:
                for _, row in cross_recs.iterrows():
                    d = row.get('domain', other_domain)
                    if d in result:
                        result[d].append(extract_details(row, d, row.get('similarity_score')))
        return result

    def _get_context_scores(self, user_id: int, candidate_movie_ids: pd.Index) -> pd.Series:
        """Data-driven context-aware scores based on user's actual genre preferences."""
        try:
            user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['UserID'] == user_id]
            if user_ratings.empty:
                return pd.Series(0.0, index=candidate_movie_ids, name='context_score')
            # Compute user's average rating per genre
            movie_genres = self.artifacts['movies']['Genres']
            genre_ratings = {}
            for _, row in user_ratings.iterrows():
                movie_id = row['MovieID']
                rating = row['Rating']
                genres = movie_genres.get(movie_id, '')
                for genre in genres.split('|'):
                    if genre:
                        genre_ratings.setdefault(genre, []).append(rating)
            # Average rating per genre
            genre_avg = {g: np.mean(r) for g, r in genre_ratings.items() if r}
            # Score each candidate movie by averaging user's avg rating for its genres
            scores = {}
            for movie_id in candidate_movie_ids:
                genres = movie_genres.get(movie_id, '')
                genre_list = [g for g in genres.split('|') if g]
                if genre_list:
                    genre_scores = [genre_avg.get(g, np.mean(list(genre_avg.values())) if genre_avg else 0.0) for g in genre_list]
                    scores[movie_id] = np.mean(genre_scores)
                else:
                    scores[movie_id] = np.mean(list(genre_avg.values())) if genre_avg else 0.0
            # Normalize scores to 0-1
            vals = np.array(list(scores.values()))
            if len(vals) > 1 and np.ptp(vals) > 0:
                minv, maxv = np.min(vals), np.max(vals)
                for k in scores:
                    scores[k] = (scores[k] - minv) / (maxv - minv)
            return pd.Series(scores, name='context_score').reindex(candidate_movie_ids).fillna(0.0)
        except Exception as e:
            self.logger.warning(f"Error computing data-driven context scores: {e}")
            return pd.Series(0.0, index=candidate_movie_ids, name='context_score')

def load_data(base_path: str) -> dict:
    return {
        'movies': {
            'users': '/content/users.dat',
            'movies': '/content/movies.dat',
            'ratings': '/content/ratings.dat'
        },
        'books': {
            'items': '/content/book1-100k.csv',
            'ratings': '/content/book1-100k.csv'
        },
        'songs': {
            'items': '/content/Music Info.csv',
            'ratings': '/content/User Listening History.csv'
        }
    }

def display_menu():
    print("\n" + "=" * 80)
    print("RECOMMENDATION SYSTEM")
    print("=" * 80)
    print("1. Show homepage")
    print("2. Get recommendations based on a movie")
    print("3. Get recommendations based on a book")
    print("4. Get recommendations based on a song")
    print("5. Get personalized recommendations")
    print("6. Exit")
    print("=" * 80)

def show_homepage(recommender: UnifiedRecommender):
    print("\n=== Homepage Sections ===\n")
    sections = recommender.get_homepage_sections()
    if not sections:
        print("No sections available")
        return
    for section_name, items in sections.items():
        print(f"\n{section_name}:")
        for idx, (_, item) in enumerate(items.iterrows(), 1):
            print(f"\n{idx}. {item['Title']}")
            print(f"   Genres: {item['Genres']}")
            if 'avg_rating' in item:
                print(f"   Rating: {item['avg_rating']:.2f}")

def get_item_recommendations(recommender: UnifiedRecommender, item_name: str, domain: str) -> None:
    try:
        if domain == 'movies':
            items = recommender.artifacts['movies']
            title_col = 'Title'
        else:
            # Robustly get items DataFrame for books/songs
            items = None
            if hasattr(recommender, 'domain_data') and domain in recommender.domain_data and 'items' in recommender.domain_data[domain]:
                items = recommender.domain_data[domain]['items']
                print(f"DEBUG: Using domain_data['{domain}']['items'] for matching.")
            elif domain == 'books' and hasattr(recommender, 'books'):
                items = recommender.books
                print("DEBUG: Using recommender.books for matching.")
            elif domain == 'songs' and hasattr(recommender, 'music_info'):
                items = recommender.music_info
                print("DEBUG: Using recommender.music_info for matching.")
            else:
                raise KeyError(f"No items DataFrame found for domain '{domain}'")
            # Try to find a title-like column
            for col in ['Title', 'Name', 'name', 'title']:
                if col in items.columns:
                    title_col = col
                    break
            else:
                title_col = items.columns[0]  # fallback to first column

        matches = items[items[title_col].str.contains(item_name, case=False, na=False)]
        if matches.empty:
            print(f"\nNo {domain} found matching: {item_name}")
            return
        print("\nFound matches:")
        for idx, (_, item) in enumerate(matches.head().iterrows(), 1):
            print(f"{idx}. {item[title_col]}")
        selection = input("\nSelect number (1-5) or press Enter for first match: ").strip()
        if not selection:
            selected_idx = 0
        else:
            try:
                selected_idx = int(selection) - 1
                if selected_idx < 0 or selected_idx >= len(matches):
                    print("Invalid selection")
                    return
            except ValueError:
                print("Invalid input")
                return
        selected_item = matches.iloc[selected_idx]
        print(f"\nRecommendations based on '{selected_item[title_col]}':\n")
        # Determine the item_id for unified/cross-domain recs
        if domain == 'movies':
            item_id = str(selected_item['MovieID']) if 'MovieID' in selected_item else str(selected_item.name)
        elif domain == 'books':
            item_id = str(selected_item['Id']) if 'Id' in selected_item else str(selected_item.name)
        elif domain == 'songs':
            item_id = str(selected_item['track_id']) if 'track_id' in selected_item else str(selected_item.name)
        else:
            item_id = str(selected_item.name)
        print(f"DEBUG: Looking for id '{item_id}' in domain '{domain}'")
        # Get recommendations from all domains
        domain_list = ['movies', 'books', 'songs']
        # Map to display names
        display_names = {'movies': 'Movies', 'books': 'Books', 'songs': 'Songs'}
        # Get recommendations for the selected domain
        if domain == 'movies':
            main_recs = recommender.get_movie_based_recommendations(item_id, top_n=10)
        else:
            main_recs = recommender.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=10)
        # Get 5 recs from each of the other domains
        cross_recs = {}
        for other_domain in domain_list:
            if other_domain == domain:
                continue
            cross_recs[other_domain] = recommender.get_cross_domain_recommendations(str(item_id), domain, n_recommendations=5)
        # Print main domain recommendations
        print(f"--- Recommended {display_names[domain]} ---")
        if main_recs is not None and not main_recs.empty:
            for idx, item in enumerate(main_recs.itertuples(), 1):
                # Try to find a title-like column for display
                for col in ['Title', 'Name', 'name', 'title']:
                    if hasattr(item, col):
                        display_title = getattr(item, col)
                        break
                else:
                    display_title = str(getattr(item, 'unified_id', item.Index))
                print(f"{idx}. {display_title}")
                print(f"   Type: {getattr(item, 'domain', display_names[domain])}")
                print(f"   Similarity Score: {getattr(item, 'similarity_score', 0):.3f}\n")
        else:
            print("Could not generate recommendations for this category.")
        # Print cross-domain recommendations
        for other_domain in cross_recs:
            print(f"--- Recommended {display_names[other_domain]} ---")
            recs = cross_recs[other_domain]
            if recs is not None and not recs.empty:
                for idx, item in enumerate(recs.itertuples(), 1):
                    for col in ['Title', 'Name', 'name', 'title']:
                        if hasattr(item, col):
                            display_title = getattr(item, col)
                            break
                    else:
                        display_title = str(getattr(item, 'unified_id', item.Index))
                    print(f"{idx}. {display_title}")
                    print(f"   Type: {getattr(item, 'domain', display_names[other_domain])}")
                    print(f"   Similarity Score: {getattr(item, 'similarity_score', 0):.3f}\n")
            else:
                print("Could not generate recommendations for this category.")
    except Exception as e:
        print(f"An error occurred while getting recommendations: {e}")

def get_personalized_recommendations(recommender: UnifiedRecommender, user_id: int):
    recommendations = recommender.get_recommendations(user_id)
    if recommendations is not None and not recommendations.empty:
        print(f"\nPersonalized recommendations for User {user_id}:")
        for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {movie['Title']}")
            print(f"   Genres: {movie['Genres']}")
            print(f"   Score: {movie['final_score']:.3f}")
    else:
        print("Could not generate recommendations for this user")

def main():
    try:
        print("Initializing Unified Recommendation System...")
        import sys
        if hasattr(sys, 'argv') and sys.argv[0]:
            current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        else:
            current_dir = os.getcwd()
        data_paths = load_data(current_dir)
        recommender = UnifiedRecommender(
            data_paths=data_paths,
            artifacts_path='./recommender_artifacts_unified',
            als_factors=100,
            als_iterations=20,
            enable_logging=True
        )
        while True:
            display_menu()
            choice = input("Enter your choice (1-6): ")
            if choice == '1':
                show_homepage(recommender)
            elif choice == '2':
                movie_name = input("\nEnter movie name: ")
                get_item_recommendations(recommender, movie_name, 'movies')
            elif choice == '3':
                book_name = input("\nEnter book name: ")
                get_item_recommendations(recommender, book_name, 'books')
            elif choice == '4':
                song_name = input("\nEnter music name: ")
                get_item_recommendations(recommender, song_name, 'songs')
            elif choice == '5':
                try:
                    user_id = int(input("\nEnter user ID: "))
                    get_personalized_recommendations(recommender, user_id)
                except ValueError:
                    print("Invalid user ID")
            elif choice == '6':
                print("\nThank you for using the Recommendation System!")
                break
            else:
                print("\nInvalid choice. Please try again.")
            input("\nPress Enter to continue...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
