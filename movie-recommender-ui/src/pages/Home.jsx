
import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import MovieCard from '../components/MovieCard';
import HeroCarousel from '../components/HeroCarousel';
import HorizontalMovieRow from '../components/HorizontalMovieRow';

export default function Home({ likedMovies, mood, time }) {
  const [recommendations, setRecommendations] = useState([]);
  const [heroMovies, setHeroMovies] = useState([]);
  const [popularMovies, setPopularMovies] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const profile = JSON.parse(localStorage.getItem("profile") || "{}");


  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch hero movies (5 featured movies)
        const heroResponse = await axios.get('http://localhost:8000/popular', {
          params: { top_k: 5 }
        });
        setHeroMovies(heroResponse.data);

        // Fetch popular movies (20 movies)
        const popularResponse = await axios.get('http://localhost:8000/popular', {
          params: { top_k: 20 }
        });
        console.log("Popular movies response:", popularResponse.data);
        setPopularMovies(popularResponse.data);

        // Fetch personalized recommendations (40 movies)
        if (likedMovies.length > 0 && mood && time) {
          const recResponse = await axios.get('http://localhost:8000/recommend/user', {
            params: {
              liked: likedMovies.join(','),
              mood,
              timeofday: time,
              top_k: 40,
              location: profile.location,
              age: profile.age

            },
          });

          setRecommendations(recResponse.data);
          console.log("Recommendations received:", recResponse.data);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [likedMovies, mood, time]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading your recommendations...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Hero Carousel */}
      <HeroCarousel movies={heroMovies} />

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8 space-y-12">
        {/* Personalized Recommendations */}
        {recommendations.length > 0 && (
          <HorizontalMovieRow 
            title="Recommended For You" 
            movies={recommendations.slice(0, 20)} 
          />
        )}

        {/* Popular Movies */}
        <HorizontalMovieRow 
          title="Popular Movies" 
          movies={popularMovies} 
        />

        {/* Genre Sections */}
        {['Action', 'Comedy', 'Drama', 'Sci-Fi'].map(genre => (
          <HorizontalMovieRow
            key={genre}
            title={`${genre} Movies`}
            movies={recommendations.filter(m => m.genres?.includes(genre)).slice(0, 20)}
          />
        ))}
      </div>
    </div>
  );
}