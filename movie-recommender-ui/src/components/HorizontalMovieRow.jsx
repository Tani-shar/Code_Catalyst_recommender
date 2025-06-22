import { motion } from 'framer-motion';
import MovieCard from './MovieCard';
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import { useRef } from 'react';

export default function HorizontalMovieRow({ title, movies = [] }) {
  const scrollRef = useRef(null);

  const scroll = (direction) => {
    if (scrollRef.current) {
      const scrollAmount = direction === 'left' ? -600 : 600;
      scrollRef.current.scrollBy({ 
        left: scrollAmount, 
        behavior: 'smooth' 
      });
    }
  };

  if (!movies || movies.length === 0) return null;

return (
    <section className="relative group">
        <div className="flex items-center justify-between mb-4 px-4">
            <h2 className="text-xl md:text-2xl font-bold">{title}</h2>
            <button className="text-blue-400 hover:text-blue-300 text-sm font-semibold">
                See All
            </button>
        </div>
        
        <div className="relative">
            <button 
                onClick={() => scroll('left')}
                className="absolute left-2 top-1/2 -translate-y-1/2 z-10 bg-gray-900/80 hover:bg-gray-700 rounded-full p-2 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg"
                aria-label={`Scroll ${title} left`}
            >
                <FiChevronLeft className="text-white text-xl md:text-2xl" />
            </button>

            <div
                ref={scrollRef}
                className="flex overflow-x-auto pb-8 space-x-4 px-4 scrollbar-hide"
                style={{
                    scrollbarWidth: 'none',
                    msOverflowStyle: 'none', 
                }}
            >
                <style>
                    {`
                        .scrollbar-hide::-webkit-scrollbar {
                            display: none;
                        }
                    `}
                </style>
                {movies.map((movie, index) => (
                    <motion.div
                        key={movie.id || index}
                        className="flex-shrink-0 w-48"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                    >
                        <MovieCard movie={movie} />
                    </motion.div>
                ))}
            </div>

            <button 
                onClick={() => scroll('right')}
                className="absolute right-2 top-1/2 -translate-y-1/2 z-10 bg-gray-900/80 hover:bg-gray-700 rounded-full p-2 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg"
                aria-label={`Scroll ${title} right`}
            >
                <FiChevronRight className="text-white text-xl md:text-2xl" />
            </button>
        </div>
    </section>
);
}