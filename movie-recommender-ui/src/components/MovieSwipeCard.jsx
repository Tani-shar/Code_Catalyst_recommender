import { Heart, X } from 'lucide-react';
import movie1 from '../assets/movie1.png';
import movie2 from '../assets/movie2.png';
import movie3 from '../assets/movie3.png';
import movie4 from '../assets/movie4.png';

// Create an array of your local images
const localMovieImages = [movie1, movie2, movie3, movie4];
const getRandomLocalImage = () => {
  const randomIndex = Math.floor(Math.random() * localMovieImages.length);
  return localMovieImages[randomIndex];
};

export default function MovieSwipeCard({ movie }) {
  console.log(movie); // Check the actual structure in console
  const imageSrc = getRandomLocalImage();
  return (
    <div className="w-80 h-[500px] bg-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col">
      <div className="h-3/4 bg-gray-600 relative">
        {(() => {
          // const imageUrl = `https://via.placeholder.com/320x480?text=${encodeURIComponent(movie.primaryTitle || movie.title)}`;
          return (
            <img
              src={movie.poster ||imageSrc}
              alt={movie.primaryTitle || movie.title}
              className="w-full h-full object-cover"
            />
          );
        })()}
      </div>
      <div className="p-4 text-white flex-grow">
        <h3 className="text-lg font-semibold truncate">{movie.primaryTitle || movie.title}</h3>
        <p className="text-sm">Rating: {movie.averageRating || movie.rating || 'N/A'}</p>
        <p className="text-sm">
          Genres: {Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres || 'N/A'}
        </p>
        <p className="text-sm">Year: {movie.startYear || movie.year || 'N/A'}</p>
        <p className="text-sm">Runtime: {movie.runtimeMinutes || movie.duration || 'N/A'} mins</p>
      </div>
    </div>
  );
}