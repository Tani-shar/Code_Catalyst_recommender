import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import MovieCard from '../components/MovieCard';

export default function MovieDetail() {
  const { title } = useParams();
  const [movie, setMovie] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // In a real app, you would fetch the specific movie details
        const movieResponse = await axios.get(`http://localhost:8000/movie/${encodeURIComponent(title)}`);
        setMovie(movieResponse.data);

        // Fetch recommendations based on this movie
        const recResponse = await axios.get('http://localhost:8000/recommend', {
          params: { movie: title, top_k: 6 }
        });
        setRecommendations(recResponse.data);
      } catch (error) {
        console.error('Error fetching movie:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [title]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading movie details...</div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Movie not found</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Hero Section */}
      <div className="relative h-[60vh] w-full">
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent z-10" />
        <img
          src={`https://image.tmdb.org/t/p/original${movie.backdrop_path}` || 
               `https://via.placeholder.com/1920x1080?text=${movie.title}`}
          alt={movie.title}
          className="w-full h-full object-cover"
        />
        
        <div className="absolute bottom-0 left-0 right-0 z-20 container mx-auto px-4 pb-12">
          <motion.h1 
            className="text-4xl md:text-5xl font-bold mb-2"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
          >
            {movie.title}
          </motion.h1>
          <div className="flex items-center gap-4 mb-4">
            <span className="text-yellow-400 font-bold">{movie.rating?.toFixed(1)}</span>
            <span>{movie.year || movie.startYear}</span>
            <span>{movie.runtimeMinutes || movie.duration}</span>
          </div>
          <p className="max-w-2xl mb-6">{movie.overview || movie.description}</p>
          <div className="flex gap-4">
            <button className="bg-blue-600 text-white px-6 py-2 rounded-md font-semibold hover:bg-blue-700 transition">
              Play Now
            </button>
            <button className="bg-gray-700 text-white px-6 py-2 rounded-md font-semibold hover:bg-gray-600 transition">
              Add to List
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* More Info */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">About</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="md:col-span-2">
              <h3 className="text-xl font-semibold mb-2">Synopsis</h3>
              <p className="text-gray-300">{movie.overview || movie.description || 'No synopsis available.'}</p>
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-2">Details</h3>
              <div className="space-y-2 text-gray-300">
                <p><span className="font-medium">Genres:</span> {movie.genres?.join(', ') || 'N/A'}</p>
                <p><span className="font-medium">Rating:</span> {movie.rating?.toFixed(1) || 'N/A'}/10</p>
                <p><span className="font-medium">Year:</span> {movie.year || movie.startYear || 'N/A'}</p>
                <p><span className="font-medium">Duration:</span> {movie.runtimeMinutes || movie.duration || 'N/A'}</p>
              </div>
            </div>
          </div>
        </section>

        {/* Recommendations */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">More Like This</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {recommendations.map((recMovie, index) => (
              <MovieCard key={index} movie={recMovie} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}