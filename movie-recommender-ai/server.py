# from fastapi import FastAPI, Query
# from recommender.utils import load_data
# from recommender.model import ContentRecommender
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()
# df = load_data()
# model = ContentRecommender(df)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to your frontend domain later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/recommend")
# def recommend(movie: str, top_k: int = 5):
#     results = model.recommend(movie, top_k)
#     return results.to_dict(orient="records")

# @app.get("/recommend/user")
# def user_recommend(
#     liked: str = Query(..., description="Comma-separated liked movie titles"),
#     mood: str = Query(None),
#     timeofday: str = Query(None),
#     top_k: int = 10
# ):
#     liked_movies = [title.strip() for title in liked.split(",")]
#     result = model.recommend_for_user(liked_movies, top_k=top_k, mood=mood, timeofday=timeofday)
#     return result.to_dict(orient="records")

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender.utils import load_data
from recommender.model import ContentRecommender
from pydantic import BaseModel
from typing import List, Optional
import recommender.songrecommender as song_model
import recommender.bookrecommender as book_model
import pandas as pd
import pickle
import os

app = FastAPI(title="Movie Recommender API", version="1.0.0")
songs_df = pd.read_csv("data/dataset.csv")  # Make sure this path is correct
books_df = pd.read_csv("data/goodreads_data.csv")  # Make sure this path is correct

song_model.song_model = song_model.SongRecommender(songs_df)
book_model.book_model = book_model.BookRecommender(books_df)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Restrict to frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache dataset to avoid reloading
CACHE_DIR = "cache"
DATA_CACHE = os.path.join(CACHE_DIR, "dataset.pkl")

if os.path.exists(DATA_CACHE):
    with open(DATA_CACHE, "rb") as f:
        df = pickle.load(f)
else:
    df = load_data()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(DATA_CACHE, "wb") as f:
        pickle.dump(df, f)

# Initialize recommender
model = ContentRecommender(df, cache_dir=CACHE_DIR)

class OnboardingRequest(BaseModel):
    liked_tconsts: List[str] = []
    disliked_tconsts: List[str] = []
    top_k: int = 20


# Pydantic models for request validation
class RecommendationRequest(BaseModel):
    movie: str
    mood: Optional[str] = None
    timeofday: Optional[str] = None
    titleType: Optional[str] = None
    isAdult: Optional[bool] = False
    top_k: Optional[int] = 5

class UserRecommendationRequest(BaseModel):
    liked: List[str]
    ratings: Optional[List[float]] = None
    mood: Optional[str] = None
    timeofday: Optional[str] = None
    titleType: Optional[str] = None
    isAdult: Optional[bool] = False
    top_k: Optional[int] = 10

class AutocompleteRequest(BaseModel):
    query: str
    top_n: Optional[int] = 10

class PopularRequest(BaseModel):
    top_k: Optional[int] = 50
    titleType: Optional[str] = None
    isAdult: Optional[bool] = False

@app.get("/popular", response_model=List[dict])
async def get_popular(
    top_k: int = Query(50, ge=1, le=100, description="Number of popular movies to return"),
    titleType: Optional[str] = Query(None, description="Filter by title type (e.g., 'movie', 'short')"),
    isAdult: bool = Query(False, description="Include adult content if True")
):
    """
    Fetch popular movies for the swipe interface.
    """
    try:
        result = model.get_popular(top_k=top_k, titleType=titleType, isAdult=isAdult)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular movies: {str(e)}")

@app.get("/autocomplete", response_model=List[dict])
async def autocomplete(
    query: str = Query(..., min_length=2, description="Search query for movie titles"),
    top_n: int = Query(10, ge=1, le=20, description="Number of suggestions to return")
):
    """
    Provide autocomplete suggestions for movie titles.
    """
    try:
        suggestions = model.get_movie_suggestions(query, top_n=top_n)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching suggestions: {str(e)}")

@app.get("/recommend", response_model=List[dict])
async def recommend(
    movie: str = Query(..., description="Movie title to base recommendations on"),
    top_k: int = Query(5, ge=1, le=50, description="Number of recommendations to return"),
    mood: Optional[str] = Query(None, description="User's mood (e.g., 'happy', 'romantic')"),
    timeofday: Optional[str] = Query(None, description="Time of day (e.g., 'morning', 'night')"),
    titleType: Optional[str] = Query(None, description="Filter by title type (e.g., 'movie', 'short')"),
    isAdult: bool = Query(False, description="Include adult content if True"),
    location: Optional[str] = Query(None),
    age: Optional[int] = Query(None)
):
    """
    Recommend movies based on a single movie title.
    """
    try:
        results = model.recommend(
            movie_title=movie,
            top_k=top_k,
            mood=mood,
            timeofday=timeofday,
            titleType=titleType,
            isAdult=isAdult,
            location=location,
            age=age
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
@app.get("/suggest")
def suggest_movies(query: str = Query(...)):
    try:
        suggestions = model.get_movie_suggestions(query)

        # Sort to prioritize exact title matches (case insensitive)
        exact_matches = [s for s in suggestions if s["title"].lower() == query.lower()]
        others = [s for s in suggestions if s["title"].lower() != query.lower()]

        ordered = exact_matches + others
        return ordered  # ✅ they already have "id", no need to remap
        
    except Exception as e:
        print("Error in /suggest route:", str(e))
        raise HTTPException(status_code=500, detail="Internal error in autocomplete suggestion.")


@app.get("/recommend/user", response_model=List[dict])
async def user_recommend(
    liked: str = Query(..., description="Comma-separated liked movie titles"),
    ratings: Optional[str] = Query(None, description="Comma-separated ratings for liked movies"),
    mood: Optional[str] = Query(None, description="User's mood (e.g., 'happy', 'romantic')"),
    timeofday: Optional[str] = Query(None, description="Time of day (e.g., 'morning', 'night')"),
    titleType: Optional[str] = Query(None, description="Filter by title type (e.g., 'movie', 'short')"),
    isAdult: bool = Query(False, description="Include adult content if True"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    location: Optional[str] = Query(None),
    age: Optional[int] = Query(None)
):
    """
    Recommend movies based on a list of liked movies.
    """
    try:
        liked_movies = [title.strip() for title in liked.split(",") if title.strip()]
        if not liked_movies:
            raise HTTPException(status_code=400, detail="At least one liked movie is required")
        
        ratings_list = None
        if ratings:
            try:
                ratings_list = [float(r) for r in ratings.split(",") if r.strip()]
                if len(ratings_list) != len(liked_movies):
                    raise ValueError("Number of ratings must match number of liked movies")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid ratings format: {str(e)}")

        result = model.recommend_for_user(
            liked_movies=liked_movies,
            ratings=ratings_list,
            mood=mood,
            timeofday=timeofday,
            titleType=titleType,
            isAdult=isAdult,
            top_k=top_k,
            location=location,
            age=age

        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating user recommendations: {str(e)}")
    
@app.get("/recommend/id", response_model=List[dict])
async def recommend_by_id(
    tconst: str = Query(..., description="IMDb ID of the movie (e.g., 'tt1375666')"),
    top_k: int = Query(5, ge=1, le=50),
    mood: Optional[str] = None,
    timeofday: Optional[str] = None,
    titleType: Optional[str] = None,
    isAdult: bool = False,
    location: Optional[str] = Query(None),
    age: Optional[int] = Query(None)

):
    """
    Recommend movies based on IMDb ID (tconst).
    """
    row = model.df[model.df['tconst'] == tconst]
    if row.empty:
        raise HTTPException(status_code=404, detail="Movie ID not found")
    
    title = row.iloc[0]['primaryTitle']
    
    try:
        result = model.recommend(
            movie_title=title,
            top_k=top_k,
            mood=mood,
            timeofday=timeofday,
            titleType=titleType,
            isAdult=isAdult,
            location=location,
            age=age
        )
        # original_movie = row.iloc[0].to_dict()
        original_movie_df = row[['tconst', 'primaryTitle', 'averageRating', 'genres', 'startYear', 'runtimeMinutes']].copy()
        original_movie_df['final_score'] = 1.0
        original_movie_df['reason'] = 'Selected movie'
        original_movie_df['genres'] = original_movie_df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
        enriched_original = model.enrich_and_return(original_movie_df)
        
        # Return the enriched original movie plus recommendations
        return enriched_original + result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/mood-based-movies", response_model=List[dict])
async def get_mood_based_movies(
    mood: str = Query(..., description="User's mood (e.g., 'happy', 'romantic')"),
    timeofday: Optional[str] = Query(None, description="Time of day (e.g., 'morning', 'night')"),
    top_k: int = Query(50, ge=1, le=100, description="Number of movies to return"),
    titleType: Optional[str] = Query(None, description="Filter by title type (e.g., 'movie', 'short')"),
    isAdult: bool = Query(False, description="Include adult content if True")
):
    """
    Fetch movies tailored to the user's mood, with a mix of relevant and varied genres.
    """
    try:
        result = model.get_mood_based_movies(
            mood=mood,
            timeofday=timeofday,
            top_k=top_k,
            titleType=titleType,
            isAdult=isAdult
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching mood-based movies: {str(e)}")

@app.get("/recommend/songs", response_model=List[dict])
def recommend_songs(
    liked: str = Query(..., description="Comma-separated liked movie titles"),
    top_k: int = Query(10, ge=1, le=50)
):
    liked_movies = [t.strip() for t in liked.split(",") if t.strip()]
    if not liked_movies:
        raise HTTPException(status_code=400, detail="At least one liked movie is required")

    main_title = liked_movies[-1]  # Use last liked movie
    return song_model.recommend_by_title(main_title, top_k=top_k)


@app.get("/recommend/books", response_model=List[dict])
def recommend_books(
    liked: str = Query(..., description="Comma-separated liked movie titles"),
    top_k: int = Query(10, ge=1, le=50)
):
    liked_movies = [t.strip() for t in liked.split(",") if t.strip()]
    if not liked_movies:
        raise HTTPException(status_code=400, detail="At least one liked movie is required")

    main_title = liked_movies[-1]  # Use last liked movie
    return book_model.recommend_by_title(main_title, top_k=top_k)

@app.get("/recommend/related", response_model=dict)
def recommend_related(title: str = Query(...), top_k: int = 5, mood: Optional[str] = None):
    try:
        # Step 1: Get similar movies
        related_movies = model.recommend(movie_title=title, top_k=top_k)
        movie_titles = [m.get("primaryTitle") for m in related_movies if m.get("primaryTitle")]

        all_songs = []
        all_books = []

        for movie in related_movies:
            mt = movie.get("primaryTitle")
            genres = movie.get("genres", [])
            overview = movie.get("overview", "")

            songs = song_model.song_model.recommend(
                movie_title=mt,
                genres=genres,
                overview=overview,
                mood=mood,
                top_k=1
            )
            all_songs.extend(songs)

            books = book_model.book_model.recommend(
                movie_title=mt,
                genres=genres,
                overview=overview,
                mood=mood,
                top_k=1
            )

            all_books.extend(books)

        return {
            "base_movie": title,
            "recommended_movies": related_movies,
            "songs": all_songs[:top_k],
            "books": all_books[:top_k]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/recommend/onboarding", response_model=List[dict])
async def onboarding_recommendations(request: OnboardingRequest):
    """
    Takes a new user's initial choices, finds the best matching
    existing user profile, and returns powerful hybrid recommendations.
    """
    if not request.liked_tconsts and not request.disliked_tconsts:
        raise HTTPException(status_code=400, detail="Must provide at least one movie choice.")

    # Find the 'fake user' ID that best matches the new user's taste
    best_user_id = model.find_best_matching_user(
        liked_tconsts=request.liked_tconsts,
        disliked_tconsts=request.disliked_tconsts
    )
    if best_user_id is None:
        print("Fallback: Could not find a matching user, returning popular movies.")
        return model.get_popular(top_k=request.top_k)
        
    print(f"Onboarding: Found best matching profile: MovieLens User #{best_user_id}")

    # Get recommendations for that 'fake user', leveraging the full hybrid model
    recommendations = model.recommend_hybrid(user_id=best_user_id, top_k=request.top_k)
    return recommendations
