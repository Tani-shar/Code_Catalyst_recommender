import { useState, useEffect } from "react";
import axios from "axios";
import MovieSwipeCard from "../components/MovieSwipeCard";
import MoodSelectorModal from "../components/MoodSelectorModal";
import TimeSelectorModal from "../components/TimeSelectorModal";
import { Heart, X } from "lucide-react";

export default function SwipePage({
  likedMovies,
  setLikedMovies,
  setMood,
  setTime,
  mood,
  time,
}) {
  const [movies, setMovies] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showMoodModal, setShowMoodModal] = useState(false);
  const [showTimeModal, setShowTimeModal] = useState(false);
  //   const [mood, setMood] = useState("");
  //   const [time, setTime] = useState("");

  useEffect(() => {
    const fetchPopularMovies = async () => {
      try {
        const response = await axios.get("http://localhost:8000/popular", {
          params: { top_k: 100 },
        });
        const allPopular = response.data;

        // Randomly shuffle and take 20
        const sampled = allPopular.sort(() => Math.random() - 0.5).slice(0, 20);

        setMovies(sampled);
      } catch (error) {
        console.error("Error fetching movies:", error);
        setMovies([
          { title: "Inception", rating: 8.8, genres: ["Sci-Fi", "Action"] },
          { title: "The Matrix", rating: 8.7, genres: ["Sci-Fi", "Action"] },
          { title: "Titanic", rating: 7.8, genres: ["Romance", "Drama"] },
        ]);
      }
    };
    fetchPopularMovies();
  }, []);

  useEffect(() => {
    console.log("Mood selected:", mood);
  }, [mood]);

  useEffect(() => {
    console.log("Time selected:", time);
  }, [time]);

  // const handleLike = () => {
  //   const movie = movies[currentIndex];
  //   if (movie && !likedMovies.includes(movie.primaryTitle || movie.title)) {
  //     setLikedMovies([...likedMovies, movie.primaryTitle || movie.title]);
  //   }
  //   console.log(likedMovies);
  //   goToNext();
  // };

  const handleLike = () => {
    const movie = movies[currentIndex];
    if (!movie) return;

    const newLikedMovie = {
      id: movie.tconst,
      poster: movie.poster || "assets/movie1.png",
      title: movie.primaryTitle || movie.title,
      stars: movie.stars || [],
      rating: movie.averageRating || "N/A",
      year: movie.startYear || "",
    };

    // const alreadyLiked = likedMovies.some((m) => m.id === newLikedMovie.id);
    if (movie && !likedMovies.includes(movie.primaryTitle || movie.title)) {
      setLikedMovies([...likedMovies, newLikedMovie]);
    }

    goToNext();
  };

  const handleDislike = () => {
    goToNext();
  };

  const goToNext = () => {
    if (likedMovies.length + 1 === 9) {
      setShowMoodModal(true);
    }

    setCurrentIndex((prev) => prev + 1);
  };

  const currentMovie = movies[currentIndex];

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold text-white mb-4">
        Select Your Favorite Movies (10)
      </h1>

      {currentMovie ? (
        <>
          <MovieSwipeCard movie={currentMovie} />
          <div className="flex space-x-6 mt-4">
            <button
              onClick={handleDislike}
              className="bg-gray-600 hover:bg-gray-500 text-white rounded-full p-4"
            >
              <X size={24} />
            </button>
            <button
              onClick={handleLike}
              className="bg-red-500 hover:bg-red-600 text-white rounded-full p-4"
            >
              <Heart size={24} />
            </button>
          </div>
          <p className="text-white mt-4">Liked: {likedMovies.length}/10</p>
          <p className="text-white mt-2">Mood: {mood}</p>
          <p className="text-white">Time: {time}</p>
        </>
      ) : (
        <p className="text-white text-lg">No more movies!</p>
      )}

      <MoodSelectorModal
        isOpen={showMoodModal}
        onClose={() => {
          setShowMoodModal(false);
          setShowTimeModal(true);
        }}
        setMood={setMood}
      />

      <TimeSelectorModal
        isOpen={showTimeModal}
        onClose={() => setShowTimeModal(false)}
        setTime={setTime}
      />
    </div>
  );
}
