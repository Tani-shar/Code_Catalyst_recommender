import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { FiUser, FiCalendar, FiMapPin, FiSave, FiTrash2, FiChevronRight, FiChevronLeft, FiEye } from "react-icons/fi";
import { motion } from "framer-motion";
import movie1 from '../assets/movie1.png';
import movie2 from '../assets/movie2.png';
import movie3 from '../assets/movie3.png';
import movie4 from '../assets/movie4.png';

export default function ProfilePage() {
  const [profile, setProfile] = useState({
    name: "",
    age: "",
    location: "",
  });
  const [likedMovies, setLikedMovies] = useState([]);
  const navigate = useNavigate();
  const scrollRef = useRef(null);
  
  const localMovieImages = [movie1, movie2, movie3, movie4];
  const getRandomLocalImage = () => {
    const randomIndex = Math.floor(Math.random() * localMovieImages.length);
    return localMovieImages[randomIndex];
  };
  // console.log(likedMovies);

  // Function to load liked movies from localStorage
  const loadLikedMovies = () => {
    const storedLikes = JSON.parse(localStorage.getItem("likedMovies")) || [];
    // console.log("Loaded liked movies:", storedLikes);
    setLikedMovies(storedLikes);
  };

  useEffect(() => {
    // Initial load of profile and liked movies
    const storedProfile = JSON.parse(localStorage.getItem("profile")) || {
      name: "",
      age: "",
      location: ""
    };
    setProfile(storedProfile);
    loadLikedMovies();

    // Listen for storage events to detect changes in localStorage
    const handleStorageChange = (event) => {
      if (event.key === "likedMovies") {
        loadLikedMovies();
      }
    };

    window.addEventListener("storage", handleStorageChange);

    // Optional: Poll localStorage periodically to handle same-tab updates
    const interval = setInterval(() => {
      loadLikedMovies();
    }, 1000); // Check every 1 second

    // Cleanup event listener and interval on component unmount
    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfile((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    localStorage.setItem("profile", JSON.stringify(profile));
    alert("Profile saved successfully!");
  };

  const handleRemove = (id) => {
    const updated = likedMovies.filter((movie) => movie.id !== id);
    setLikedMovies(updated);
    localStorage.setItem("likedMovies", JSON.stringify(updated));
  };

  const scroll = (direction) => {
    if (scrollRef.current) {
      const scrollAmount = direction === 'left' ? -300 : 300;
      scrollRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 py-12 px-2 sm:px-8 lg:px-16">
      <div className="max-w-7xl mx-auto space-y-10">
        {/* Profile Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800/50 backdrop-blur-lg rounded-xl shadow-2xl overflow-hidden w-full"
        >
          <div className="p-10">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-2">
              <FiUser className="text-blue-400" />
              User Profile
            </h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center bg-gray-700/50 rounded-lg px-4 py-3 border border-gray-600/50">
                  <FiUser className="text-gray-400 mr-3" />
                  <input
                    name="name"
                    value={profile.name}
                    onChange={handleChange}
                    placeholder="Your Name"
                    className="w-full bg-transparent text-white placeholder-gray-400 focus:outline-none"
                  />
                </div>
                
                <div className="flex items-center bg-gray-700/50 rounded-lg px-4 py-3 border border-gray-600/50">
                  <FiCalendar className="text-gray-400 mr-3" />
                  <input
                    name="age"
                    type="number"
                    value={profile.age}
                    onChange={handleChange}
                    placeholder="Your Age"
                    className="w-full bg-transparent text-white placeholder-gray-400 focus:outline-none"
                  />
                </div>
                
                <div className="flex items-center bg-gray-700/50 rounded-lg px-4 py-3 border border-gray-600/50">
                  <FiMapPin className="text-gray-400 mr-3" />
                  <input
                    name="location"
                    value={profile.location}
                    onChange={handleChange}
                    placeholder="Your Location"
                    className="w-full bg-transparent text-white placeholder-gray-400 focus:outline-none"
                  />
                </div>
              </div>
              
              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-medium py-3 px-6 rounded-lg transition-all duration-300 shadow-lg"
              >
                <FiSave />
                Save Profile
              </button>
            </form>
          </div>
        </motion.div>
            
        {/* Watchlist Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-gray-800/50 backdrop-blur-lg rounded-xl shadow-2xl overflow-hidden w-full p-10"
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-bold text-white flex items-center gap-2">
              <FiEye className="text-blue-400" />
              Your Liked Movies
            </h2>
            {likedMovies.length > 4 && (
              <div className="flex gap-2">
                <button 
                  onClick={() => scroll('left')}
                  className="p-2 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                >
                  <FiChevronLeft size={18} />
                </button>
                <button 
                  onClick={() => scroll('right')}
                  className="p-2 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                >
                  <FiChevronRight size={18} />
                </button>
              </div>
            )}
          </div>

          {likedMovies.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 mb-4">Your watchlist is empty</p>
              <button 
                onClick={() => navigate('/')}
                className="text-blue-400 hover:text-blue-300 font-medium hover:underline flex items-center justify-center gap-1 mx-auto"
              >
                Browse movies <FiChevronRight className="mt-0.5" />
              </button>
            </div>
          ) : (
            <div className="relative">
              <div
                ref={scrollRef}
                className="flex overflow-x-auto pb-6 gap-6 scrollbar-hide px-1"
                style={{
                  scrollbarWidth: "none",
                  msOverflowStyle: "none"
                }}
              >
                {/* Hide scrollbar for Chrome, Safari and Opera */}
                <style>
                  {`
                    .scrollbar-hide::-webkit-scrollbar {
                      display: none;
                    }
                  `}
                </style>
                {likedMovies.map((movie) => {
                  const imagesrc = getRandomLocalImage();
                  return (
                    <motion.div
                      key={movie.id}
                      whileHover={{ scale: 1.03 }}
                      className="flex-shrink-0 w-72 bg-gray-700/50 rounded-xl overflow-hidden shadow-lg relative group border border-gray-600/50"
                    >
                      <div className="h-56 bg-gradient-to-br from-gray-800 to-gray-700">
                        <img 
                          src={movie.poster || imagesrc} 
                          alt={movie.title}
                          className="w-full h-full object-cover opacity-90"
                        />
                      </div>
                      <div className="p-4">
                        <h3 className="font-medium text-white mb-1 truncate">{movie.title}</h3>
                        <div className="flex justify-between items-center text-sm text-gray-400">
                          <span>Rating: {movie.rating || 'N/A'}</span>
                          <span>{movie.year || ''}</span>
                        </div>
                      </div>
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
                        <div className="flex gap-2 w-full">
                          <button
                            onClick={() => navigate(`/movie/${movie.id}`)}
                            className="flex-1 bg-blue-600 hover:bg-blue-500 text-white py-2 rounded text-sm font-medium transition-colors"
                          >
                            <FiEye className="inline mr-1" /> View
                          </button>
                          <button
                            onClick={() => handleRemove(movie.id)}
                            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white py-2 rounded text-sm font-medium transition-colors"
                          >
                            <FiTrash2 className="inline mr-1" /> Remove
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}