from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import your recommender class from Final.py
from Final import HybridRecommenderV3

# Define request and response schemas
typing_defs = __import__('typing').Any
class RecRequest(BaseModel):
    user_id: int
    top_n: int = 10

class RecItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    final_score: float

# Initialize FastAPI
app = FastAPI(
    title="Hybrid Recommender Service",
    description="FastAPI wrapper around HybridRecommenderV3",
    version="1.0.0"
)

# Instantiate the recommender model (adjust paths as needed)
model = HybridRecommenderV3(
    data_paths={
        'users': './data/users.dat',
        'movies': './data/movies.dat',
        'ratings': './data/ratings.dat'
    },
    artifacts_path='./recommender_artifacts_v3'
)

@app.post('/recommend', response_model=list[RecItem])
def recommend(req: RecRequest):
    """
    Generate top-N movie recommendations for a given user.
    """
    df = model.get_recommendations(req.user_id, req.top_n)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {req.user_id}")

    items = []
    for movie_id, row in df.iterrows():
        items.append(RecItem(
            movie_id=int(movie_id),
            title=row['Title'],
            genres=row['Genres'].split('|'),
            final_score=float(row['final_score'])
        ))
    return items

@app.get('/user/{user_id}')
def user_info(user_id: int):
    info = model.get_user_info(user_id)
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    return info

@app.get('/movie/{movie_id}')
def movie_info(movie_id: int):
    info = model.get_movie_info(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail="Movie not found")
    return info

@app.get('/stats')
def model_stats():
    return model.get_model_stats()

# Entry point
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
