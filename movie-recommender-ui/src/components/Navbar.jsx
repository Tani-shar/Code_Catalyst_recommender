import { useNavigate } from "react-router-dom";
import { User } from "lucide-react"; // optional: use an icon or an image
import { motion } from "framer-motion";
import SearchBar from "./Searchbar";
import { Sun, Moon } from "lucide-react"; // icons for theme toggle

export default function Navbar({ theme, toggleTheme }) {
  const navigate = useNavigate();

  return (
    <nav className="bg-black/90 backdrop-blur-md border-b border-gray-800/50 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left: Logo */}
          <motion.div 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-2"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-md flex items-center justify-center">
              <span className="text-white font-bold text-lg">C</span>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              CodeCatalyst
            </span>
          </motion.div>

          {/* Center: Search */}
          <div className="flex-grow mx-4">
            <SearchBar />
          </div>

          {/* Right: Theme Toggle + Avatar */}
          <div className="flex items-center gap-3">
            <motion.div
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => navigate("/profile")}
              className="w-9 h-9 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 cursor-pointer flex items-center justify-center text-white font-semibold"
              title="Profile"
            >
              <User size={18} />
            </motion.div>
          </div>
        </div>
      </div>
    </nav>
  );
}
