import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import axios from "axios";

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchSuggestions = async () => {
      if (query.length < 2) {
        setSuggestions([]);
        return;
      }
      try {
        const res = await axios.get("http://localhost:8000/suggest", {
          params: { query },
        });
        setSuggestions(res.data);
      } catch (error) {
        console.error("Failed to fetch suggestions:", error);
      }
    };

    const debounce = setTimeout(fetchSuggestions, 200);
    return () => clearTimeout(debounce);
  }, [query]);

  useEffect(() => {
    if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
      setQuery(suggestions[highlightIndex].title);
    }
  }, [highlightIndex]);

  const handleSelect = (movie) => {
    console.log("Navigating to:", movie.id);
    setQuery("");
    setSuggestions([]);
    setShowDropdown(false);
    setHighlightIndex(-1);
    navigate(`/movie/${movie.id}`);
  };

  return (
    <div className="relative w-full max-w-md">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setShowDropdown(true);
          setHighlightIndex(-1);
        }}
        onFocus={() => query && setShowDropdown(true)}
        onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            setHighlightIndex((prev) =>
              suggestions.length ? (prev + 1) % suggestions.length : 0
            );
          } else if (e.key === "ArrowUp") {
            setHighlightIndex((prev) =>
              suggestions.length
                ? (prev - 1 + suggestions.length) % suggestions.length
                : 0
            );
          } else if (e.key === "Enter") {
            if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
              handleSelect(suggestions[highlightIndex]);
            } else if (suggestions.length > 0) {
              handleSelect(suggestions[0]);
            }
          }
        }}
        placeholder="Search for movies..."
        className="w-full px-4 py-2 rounded-md bg-gray-800 text-white placeholder-gray-400 focus:outline-none"
      />
      {showDropdown && suggestions.length > 0 && (
        <motion.ul
          className="absolute top-full left-0 right-0 bg-gray-900 border border-gray-700 mt-1 rounded-md shadow-lg z-50"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {suggestions.map((s, index) => (
            <li
              key={s.id}
              onClick={() => handleSelect(s)}
              className={`px-4 py-2 cursor-pointer text-white ${
                highlightIndex === index ? "bg-gray-700" : "hover:bg-gray-800"
              }`}
            >
              {s.title}
            </li>
          ))}
        </motion.ul>
      )}
    </div>
  );
}
