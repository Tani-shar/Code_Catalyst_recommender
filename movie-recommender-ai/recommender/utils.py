# recommender/utils.py
import pandas as pd

def load_data():
    basics = pd.read_csv("data/title.basics.tsv", sep="\t", na_values="\\N", low_memory=False)
    ratings = pd.read_csv("data/title.ratings.tsv", sep="\t", na_values="\\N")

    basics = basics[basics["titleType"] == "movie"]
    basics = basics.dropna(subset=["primaryTitle", "genres"])

    df = pd.merge(basics, ratings, on="tconst")
    df = df.dropna(subset=["averageRating"])

    # ✅ Reduce size: take top 2000 most-voted movies
    df = df.sort_values(by="numVotes", ascending=False).head(2000).reset_index(drop=True)
    return df
