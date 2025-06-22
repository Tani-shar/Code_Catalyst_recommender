# In train_cf_model.py
import pandas as pd
from surprise import Reader, Dataset, SVD
import pickle
import os

RATINGS_FILE = 'data/user_ratings_prepared.csv'
MODEL_PATH = 'cache/cf_model.pkl'
os.makedirs('cache', exist_ok=True)

def train_and_save_model():
    print("Starting collaborative filtering model training...")
    if not os.path.exists(RATINGS_FILE):
        print(f"'{RATINGS_FILE}' not found. Please run prepare_ratings.py first. Skipping.")
        return
        
    ratings_df = pd.read_csv(RATINGS_FILE)
    reader = Reader(rating_scale=(-1, 1))
    data = Dataset.load_from_df(ratings_df[['userId', 'tconst', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=100, n_epochs=20, random_state=42)
    algo.fit(trainset)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(algo, f)
    print(f"✅ Collaborative filtering model trained and saved to '{MODEL_PATH}'")

if __name__ == '__main__':
    train_and_save_model()