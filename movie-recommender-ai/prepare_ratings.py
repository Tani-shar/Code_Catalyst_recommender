# In prepare_ratings.py
import pandas as pd
import os

def prepare_movielens_ratings():
    print("Preparing MovieLens ratings data...")
    DATA_DIR = 'data'
    
    try:
        ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please download the 'ml-latest-small' dataset and place ratings.csv and links.csv in the 'data' directory.")
        return

    links['tconst'] = links['imdbId'].apply(lambda x: f"tt{str(x).zfill(7)}")
    df = pd.merge(ratings, links, on='movieId')
    df['normalized_rating'] = df['rating'].apply(lambda x: 1.0 if x >= 3.5 else -1.0)
    
    prepared_df = df[['userId', 'tconst', 'normalized_rating']].rename(columns={'normalized_rating': 'rating'})
    
    output_path = os.path.join(DATA_DIR, 'user_ratings_prepared.csv')
    prepared_df.to_csv(output_path, index=False)
    print(f"✅ Prepared ratings file saved to '{output_path}'. Total ratings: {len(prepared_df)}")

if __name__ == '__main__':
    prepare_movielens_ratings()