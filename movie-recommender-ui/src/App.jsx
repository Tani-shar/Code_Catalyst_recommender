import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import SwipePage from "./pages/SwipePage";
import Home from "./pages/Home";
import MovieDetails from "./pages/MovieDetails";
import "./index.css";
import ProfilePage from './pages/Profile';

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");
  const [likedMovies, setLikedMovies] = useState(
    JSON.parse(localStorage.getItem("likedMovies")) || []
  );
  const [mood, setMood] = useState(localStorage.getItem("mood") || "");
  const [time, setTime] = useState(localStorage.getItem("time") || "");
  const [showMoodModal, setShowMoodModal] = useState(
    likedMovies.length === 10 && !mood
  );
  const [showTimeModal, setShowTimeModal] = useState(
    likedMovies.length === 10 && mood && !time
  );

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
    localStorage.setItem("likedMovies", JSON.stringify(likedMovies));
    localStorage.setItem("mood", mood);
    localStorage.setItem("time", time);
    if (likedMovies.length === 10 && !mood) setShowMoodModal(true);
    if (likedMovies.length === 10 && mood && !time) setShowTimeModal(true);
  }, [theme, likedMovies, mood, time]);

  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");

  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <Navbar theme={theme} toggleTheme={toggleTheme} />
        <Routes>
          <Route
            path="/"
            element={
              likedMovies.length < 10 ? (
                <SwipePage
                  likedMovies={likedMovies}
                  setLikedMovies={setLikedMovies}
                  setMood={setMood}
                  setTime={setTime}
                />
              ) : (
                <Home likedMovies={likedMovies} mood={mood} time={time} />
              )
            }
          />
          {/* <Route path="/movie/:title" element={<MovieDetails />} /> */}
          {/* <Route path="/movie/:id" element={<MovieDetails />} /> */}
          <Route path="/movie/:id" element={<MovieDetails />} />
          <Route path="/profile" element={<ProfilePage />} />

        </Routes>
      </div>
    </BrowserRouter>
  );
}
