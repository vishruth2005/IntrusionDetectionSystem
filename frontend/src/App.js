import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./components/Home";
import Generate from "./components/Generate";
import Predict from "./components/Predict";
import Capture from "./components/Capture";

const App = () => (
  <Router>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/generate" element={<Generate />} />
      <Route path="/predict" element={<Predict />} />
      <Route path="/capture" element={<Capture />} />
    </Routes>
  </Router>
);

export default App;
