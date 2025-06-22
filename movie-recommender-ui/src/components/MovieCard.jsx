import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
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

export default function MovieCard({ movie }) {
  const id = movie.tconst || movie.id || movie._id || "unknown-id";
  // console.log("Movie ID:", id); // Log the movie ID for debugging
  const title = movie.title || movie.primaryTitle || "Untitled";
  const rating = movie.rating ?? movie.averageRating ?? "N/A";
  const year = movie.year ?? movie.startYear ?? "";
  const imageSrc = getRandomLocalImage();
  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      className="relative group"
    >
      <Link to={`/movie/${movie.tconst}`}>

        <div className="relative aspect-[2/3] rounded-lg overflow-hidden shadow-lg">
          <img
            src={movie.poster || imageSrc}
            alt={title}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
            <div className="flex items-center mb-1">
              <span className="text-yellow-400 font-bold mr-1">{rating}</span>
              <span className="text-white text-sm">/10</span>
            </div>
            <div className="text-white text-sm mb-2">
              {Array.isArray(movie.genres) ? movie.genres.slice(0, 2).join(', ') : ''}
            </div>
          </div>
        </div>
        <h3 className="text-white font-medium mt-2 line-clamp-1">{title}</h3>
        {year && <p className="text-gray-400 text-sm">{year}</p>}
      </Link>
    </motion.div>
  );
}