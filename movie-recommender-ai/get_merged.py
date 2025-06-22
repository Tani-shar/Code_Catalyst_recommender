import pandas as pd

# Load IMDb TSV files from 'data' folder
basics = pd.read_csv("data/title.basics.tsv", sep='\t', na_values='\\N', dtype=str)
ratings = pd.read_csv("data/title.ratings.tsv", sep='\t', na_values='\\N', dtype=str)

# Merge the two datasets
df = basics.merge(ratings, on="tconst", how="left")

# Filter to only movies and shorts
df = df[df["titleType"].isin(["movie", "short"])]

# Keep important columns
df = df[[
    "tconst", "titleType", "primaryTitle", "originalTitle",
    "isAdult", "startYear", "runtimeMinutes", "genres", "averageRating"
]]

# Save to CSV in the same folder as your model.py
df.to_csv("movies.csv", sep="\t", index=False)

print("✅ movies.csv generated successfully!")
