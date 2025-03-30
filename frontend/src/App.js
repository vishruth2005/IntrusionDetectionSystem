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
    domain={process.env.REACT_APP_AUTH0_DOMAIN} // Use environment variable for Auth0 domain
    clientId={process.env.REACT_APP_AUTH0_CLIENT_ID} // Use environment variable for Auth0 client ID
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