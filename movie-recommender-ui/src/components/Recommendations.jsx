import React from "react";
import { motion } from "framer-motion";
import { FiChevronLeft, FiChevronRight } from "react-icons/fi";
import movie1 from "../assets/movie1.png";
import movie2 from "../assets/movie2.png";
import movie3 from "../assets/movie3.png";
import movie4 from "../assets/movie4.png";
import songPlaceholder from "../assets/song1.png"; 
import bookPlaceholder from "../assets/book1.png"; 

// Create arrays of local images
const localMovieImages = [movie1, movie2, movie3, movie4];
const getRandomLocalImage = (type = "movie") => {
  if (type === "song") return songPlaceholder;
  if (type === "book") return bookPlaceholder;
  const randomIndex = Math.floor(Math.random() * localMovieImages.length);
  return localMovieImages[randomIndex];
};

const ScrollableSection = ({ title, items, renderItem }) => {
  const scrollRef = React.useRef(null);

  const scroll = (direction) => {
    if (scrollRef.current) {
      const scrollAmount = direction === "left" ? -300 : 300;
      scrollRef.current.scrollBy({
        left: scrollAmount,
        behavior: "smooth",
      });
    }
  };

  return (
    <div className="mt-8 relative">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white">{title}</h2>
        <div className="flex gap-2">
          <button
            onClick={() => scroll("left")}
            className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            <FiChevronLeft className="text-gray-800 dark:text-gray-200" />
          </button>
          <button
            onClick={() => scroll("right")}
            className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            <FiChevronRight className="text-gray-800 dark:text-gray-200" />
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <p className="text-gray-500 dark:text-gray-400">
          No {title.toLowerCase()} yet.
        </p>
      ) : (
        <div className="relative">
          <div
            ref={scrollRef}
            className="flex overflow-x-auto pb-4 gap-4 scrollbar-hide"
          >
            {items.map(renderItem)}
          </div>
        </div>
      )}
    </div>
  );
};

export default function Recommendations({ recommendations }) {
  const { movies = [], songs = [], books = [] } = recommendations;

  const uniqueBooks = books.filter(
    (book, index, self) =>
      index ===
      self.findIndex(
        (b) =>
          b.Book?.toLowerCase().trim() === book.Book?.toLowerCase().trim() &&
          b.Author?.toLowerCase().trim() === book.Author?.toLowerCase().trim()
      )
  );
  const uniqueSongs = songs.filter(
    (song, index, self) =>
      index ===
      self.findIndex(
        (s) =>
          s.track_name?.toLowerCase().trim() ===
            song.track_name?.toLowerCase().trim() &&
          s.artists?.toLowerCase().trim() ===
            song.artists?.toLowerCase().trim()
      )
  );

  // Render movie card
  const renderMovie = (movie, index) => (
    <motion.div
      key={`movie-${index}`}
      className="flex-shrink-0 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
      onClick={() => (window.location.href = `/movie/${movie.tconst}`)}
    >
      <div className="h-96 bg-gray-300 dark:bg-gray-700">
        <img
          src={movie.poster || getRandomLocalImage("movie")}
          alt={movie.primaryTitle}
          className="w-full h-full object-cover"
        />
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg line-clamp-1 text-white">
          {movie.primaryTitle || movie.title || "Untitled"}
        </h3>
        <div className="flex items-center mt-1">
          <span className="text-yellow-500 font-medium">
            {movie.averageRating?.toFixed(1) || "N/A"}
          </span>
          <span className="text-gray-400 text-sm ml-2">
            {Array.isArray(movie.genres)
              ? movie.genres.slice(0, 2).join(", ")
              : "N/A"}
          </span>
        </div>
      </div>
    </motion.div>
  );

  // Render song card
  const renderSong = (song, index) => (
    <motion.div
      key={`song-${index}`}
      className="flex-shrink-0 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
    >
      <div className="h-64 bg-gray-300 dark:bg-gray-700">
        <img
          src={getRandomLocalImage("song")}
          alt={song.track_name}
          className="w-full h-full object-cover"
        />
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg line-clamp-1 text-white">
          {song.track_name || "Untitled Song"}
        </h3>
        <p className="text-gray-400 text-sm line-clamp-1">
          {song.artists || "Unknown Artist"}
        </p>
        <p className="text-gray-400 text-sm mt-1">
          {song.track_genre || "N/A"}
        </p>
      </div>
    </motion.div>
  );

  // Render book card
  const renderBook = (book, index) => (
    <motion.div
      key={`book-${index}`}
      className="flex-shrink-0 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
      onClick={() => window.open(book.URL, "_blank")} // Open Goodreads URL in new tab
    >
      <div className="h-96 bg-gray-300 dark:bg-gray-700">
        <img
          src={getRandomLocalImage("book")}
          alt={book.Book}
          className="w-full h-full object-cover"
        />
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg line-clamp-1 text-white">
          {book.Book || "Untitled Book"}
        </h3>
        <p className="text-gray-400 text-sm line-clamp-1">
          {book.Author || "Unknown Author"}
        </p>
        <div className="flex items-center mt-1">
          <span className="text-yellow-500 font-medium">
            {book.Avg_Rating?.toFixed(1) || "N/A"}
          </span>
          <span className="text-gray-400 text-sm ml-2">
            {book.Genres ? eval(book.Genres).slice(0, 2).join(", ") : "N/A"}
          </span>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div>
      <ScrollableSection
        title="Recommended Movies"
        items={movies}
        renderItem={renderMovie}
      />
      <ScrollableSection
        title="Recommended Songs"
        items={uniqueSongs}
        renderItem={renderSong}
      />
      <ScrollableSection
        title="Recommended Books"
        items={uniqueBooks}
        renderItem={renderBook}
      />
    </div>
  );
}
