import React, { useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { useNavigate } from "react-router-dom"; // Import useNavigate for redirection
import "./Landing.css"; // Import CSS for styling

const Landing = () => {
  const { loginWithRedirect, isAuthenticated } = useAuth0();
  const navigate = useNavigate(); // Initialize useNavigate

  useEffect(() => {
    if (isAuthenticated) {
      navigate("/home"); // Redirect to Home if authenticated
    }
  }, [isAuthenticated, navigate]);

  const handleSignUp = () => {
    // Redirect to the Auth0 sign-up page
    loginWithRedirect({ screen_hint: "signup" });
  };

  return (
    <div className="landing-container">
      <div className="overlay">
        <div className="content">
          <h1 className="slide-in">Intrusion Detection System</h1>
          <p className="slide-in">
            Safeguard your network with our advanced intrusion detection system.
            Monitor, detect, and respond to threats in real-time.
          </p>
          <div className="auth-buttons">
            <button className="login-button" onClick={() => loginWithRedirect()}>
              Login
            </button>
            <button className="signup-button" onClick={handleSignUp}>
              Sign Up
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;