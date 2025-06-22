import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import movie1 from "../assets/moviehero1.png";
import movie2 from "../assets/moviehero2.png";
import movie3 from "../assets/moviehero3.png";
const localMovieImages = [movie1, movie2, movie3];

export default function HeroCarousel({ movies }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const getRandomLocalImage = () => {
    const randomIndex = Math.floor(Math.random() * localMovieImages.length);
    return localMovieImages[randomIndex];
  };
  const imageSrc = getRandomLocalImage();

  useEffect(() => {
    if (movies.length > 1) {
      const interval = setInterval(() => {
        setCurrentIndex((prev) => (prev + 1) % movies.length);
      }, 8000);
      return () => clearInterval(interval);
    }
  }, [movies]);

  if (movies.length === 0) return null;

  const currentMovie = movies[currentIndex];

  return (
    <div className="relative h-[70vh] w-full overflow-hidden">
      <AnimatePresence mode="wait">
        <motion.div
          key={currentMovie.id}
          className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1 }}
        >
          <img
            src={imageSrc}
            alt={currentMovie.title}
            className="w-full h-full"
          />
        </motion.div>
      </AnimatePresence>

      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-900 to-transparent h-1/2">
        <div className="container mx-auto px-4 pb-12 pt-24">
          <motion.h1
            className="text-4xl md:text-5xl font-bold mb-4"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            {currentMovie.primaryTitle}
          </motion.h1>
          <motion.p
            className="text-lg max-w-2xl mb-6"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {currentMovie.overview ||
              "An exciting movie experience awaits you."}
          </motion.p>
          <motion.div
            className="flex gap-4"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7 }}
          >
            <button className="bg-white text-black px-6 py-2 rounded-md font-semibold hover:bg-opacity-90 transition">
              Play Now
            </button>
            <button className="bg-gray-600 bg-opacity-70 text-white px-6 py-2 rounded-md font-semibold hover:bg-opacity-50 transition">
              More Info
            </button>
          </motion.div>
        </div>
      </div>

      <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2">
        {movies.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            className={`w-3 h-3 rounded-full ${
              currentIndex === index ? "bg-white" : "bg-gray-500"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
