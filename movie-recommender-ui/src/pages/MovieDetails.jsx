import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { FaStar, FaPlay, FaPlus, FaClock, FaCalendarAlt } from 'react-icons/fa';
import Recommendations from '../components/Recommendations';
import movie1 from '../assets/movie1.png';
import movie2 from '../assets/movie2.png';
import movie3 from '../assets/movie3.png';
import movie4 from '../assets/movie4.png';

const localMovieImages = [movie1, movie2, movie3, movie4];
const getRandomLocalImage = () => {
  const randomIndex = Math.floor(Math.random() * localMovieImages.length);
  return localMovieImages[randomIndex];
};

export default function MovieDetails() {
  const { id } = useParams(); // id is tconst
  const [movie, setMovie] = useState(null);
  const [rating, setRating] = useState(0);
  const [recommendations, setRecommendations] = useState({
    movies: [],
    songs: [],
    books: []
  });
  const [cast, setCast] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const profile = JSON.parse(localStorage.getItem('profile')) || {};

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);

        // Fetch main movie details using tconst
        const movieResponse = await axios.get('http://localhost:8000/recommend/id', {
          params: { tconst: id, top_k: 1 }
        });
        console.log(movieResponse.data);
        const mainMovie = movieResponse.data[0] || {
          title: 'Untitled',
          rating: 'N/A',
          genres: [],
          startYear: 'N/A',
          runtimeMinutes: 'N/A',
        };
        setMovie(mainMovie);
        console.log(mainMovie);
        

        // Fetch related recommendations using primaryTitle
        const relatedResponse = await axios.get('http://localhost:8000/recommend/related', {
          params: {
            title: mainMovie.primaryTitle || mainMovie.title,
            top_k: 7,
            mood: profile.mood || 'happy',
            location: profile.location,
            age: profile.age
          }
        });

        // Extract movies, songs, and books from response
        const { recommended_movies, songs, books } = relatedResponse.data;
        setRecommendations({
          movies: recommended_movies.filter(m => m.tconst !== id).slice(0, 6), // Exclude main movie
          songs: songs.slice(0, 5),
          books: books.slice(0, 5)
        });

        // Set cast
        const stars = Array.isArray(mainMovie.stars) ? mainMovie.stars : [];
        const mockCast = stars.slice(0, 4).map((star) => ({ name: star }));
        setCast(mockCast);

        // Set rating from local storage
        const storedRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
        setRating(storedRatings[id] || 0);
      } catch (error) {
        console.error('Error fetching movie data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [id, profile.mood, profile.location, profile.age]);

  const handleRating = (value) => {
    setRating(value);
    const storedRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
    storedRatings[id] = value;
    localStorage.setItem('userRatings', JSON.stringify(storedRatings));

    const likedList = JSON.parse(localStorage.getItem('likedMovies') || '[]');
    const updatedList = likedList.filter((m) => m.id !== id);

    if (value === 1) {
      updatedList.push({
        id,
        poster: movie.poster || getRandomLocalImage(),
        title: movie.primaryTitle || movie.title,
        stars: movie.stars || [],
        rating: movie.averageRating || 'N/A',
        year: movie.startYear || ''
      });
    }

    localStorage.setItem('likedMovies', JSON.stringify(updatedList));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading movie details...</div>
      </div>
    );
  }

  const imageSrc = getRandomLocalImage();

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="relative h-24 md:h-36 w-full bg-gradient-to-r from-gray-800 to-gray-900">
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent z-10" />
        <div className="container mx-auto h-full flex items-end pb-4 relative z-20 px-4 md:px-6">
          <h1 className="text-3xl md:text-5xl font-bold drop-shadow-lg">
            {movie.primaryTitle || movie.title || 'Untitled'}
          </h1>
        </div>
      </div>

      <div className="container mx-auto px-4 md:px-6 py-10">
        <div className="flex flex-col md:flex-row gap-8 lg:gap-12">
          <div className="w-full md:w-1/3 lg:w-1/4 flex-shrink-0">
            <img
              src={movie.poster || imageSrc}
              alt={movie.primaryTitle}
              className="w-full rounded-lg shadow-xl"
            />
          </div>

          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-4 mb-6">
              <div className="flex items-center bg-gray-800 px-3 py-1 rounded-full">
                <FaStar className="text-yellow-400 mr-1" />
                <span>{movie.averageRating || 'N/A'}</span>
              </div>
              <div className="flex items-center text-gray-300">
                <FaCalendarAlt className="mr-1" />
                <span>{movie.startYear || 'N/A'}</span>
              </div>
              <div className="flex items-center text-gray-300">
                <FaClock className="mr-1" />
                <span>{movie.runtimeMinutes ? `${movie.runtimeMinutes} min` : 'N/A'}</span>
              </div>
              {movie.genres && (
                <div className="flex flex-wrap gap-2">
                  {(Array.isArray(movie.genres) ? movie.genres : movie.genres?.split(",")).slice(0, 3).map((genre) => (
                    <span key={genre} className="px-3 py-1 bg-gray-800 rounded-full text-sm">
                      {genre.trim()}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="flex gap-4 mb-8">
              <button className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-md font-semibold transition-colors">
                <FaPlay />
                Play Now
              </button>
              <button className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-6 py-3 rounded-md font-semibold transition-colors">
                <FaPlus />
                My List
              </button>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-2">Rate This Movie</h3>
              <div className="flex items-center gap-4">
                <button
                  className={`p-2 rounded-full border-2 transition-colors ${
                    rating === 1
                      ? 'bg-green-600 border-green-600 text-white'
                      : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-green-700 hover:border-green-700'
                  }`}
                  onClick={() => handleRating(1)}
                  aria-label="Thumbs Up"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path fill={rating === 1 ? 'white' : 'none'} stroke="currentColor" strokeWidth={2} d="M14 9V5a3 3 0 00-6 0v4H4a2 2 0 00-2 2v7a2 2 0 002 2h12.72a2 2 0 001.98-1.75l1.3-9A2 2 0 0018 7h-4z" />
                  </svg>
                </button>
                <button
                  className={`p-2 rounded-full border-2 transition-colors ${
                    rating === -1
                      ? 'bg-red-600 border-red-600 text-white'
                      : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-red-700 hover:border-red-700'
                  }`}
                  onClick={() => handleRating(-1)}
                  aria-label="Thumbs Down"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path fill={rating === -1 ? 'white' : 'none'} stroke="currentColor" strokeWidth={2} d="M10 15v4a3 3 0 006 0v-4h4a2 2 0 002-2v-7a2 2 0 00-2-2H5.28a2 2 0 00-1.98 1.75l-1.3 9A2 2 0 006 17h4z" />
                  </svg>
                </button>
                <span className="ml-3 text-gray-300">
                  {rating === 1
                    ? 'You liked this'
                    : rating === -1
                    ? 'You disliked this'
                    : 'Not rated yet'}
                </span>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-2">Synopsis</h3>
              <p className="text-gray-300 leading-relaxed">
                {movie.overview || 'No synopsis available.'}
              </p>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4">Cast</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                {cast.map((person, index) => (
                  <div key={index} className="bg-gray-800/50 rounded-lg p-3">
                    <h4 className="font-medium text-white">{person.name}</h4>
                    <p className="text-sm text-gray-400">{person.character}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12">
          <Recommendations recommendations={recommendations} />
        </div>
      </div>
    </div>
  );
}