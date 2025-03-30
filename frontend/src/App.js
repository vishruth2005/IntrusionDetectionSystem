import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { Auth0Provider } from "@auth0/auth0-react"; // Import Auth0Provider
import Home from "./components/Home";
import Generate from "./components/Generate";
import Predict from "./components/Predict";
import Capture from "./components/Capture";
import Landing from "./components/Landing"; // Import the Landing component

const App = () => (
  <Auth0Provider
    domain="dev-kt2cho1vhzoc36ib.us.auth0.com" // Replace with your Auth0 domain
    clientId="ZTeTqUk6y5kExOL1yGN7ZQBCYHnOzVPE" // Replace with your Auth0 client ID
    authorizationParams={{
      redirect_uri: window.location.origin // Use authorizationParams.redirect_uri instead of redirectUri
    }}
  >
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} /> {/* Update the root route */}
        <Route path="/home" element={<Home />} />
        <Route path="/generate" element={<Generate />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/capture" element={<Capture />} />
      </Routes>
    </Router>
  </Auth0Provider>
);

export default App;