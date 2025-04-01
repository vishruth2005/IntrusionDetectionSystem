import React from "react";
import { Link } from "react-router-dom";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import "./Home.css"; // Import CSS

const Home = () => {
  const { logout } = useAuth0(); // Destructure logout from useAuth0

  return (
    <div className="home-container">
      <button className="logout-button" onClick={() => logout({ returnTo: window.location.origin })}>Logout</button> {/* Add logout functionality */}
      <h1>Intrusion Detection System</h1>
      <div className="placards">
        <Link to="/generate" className="placard">
          <h2>Generate</h2>
          <p>Create synthetic datasets to simulate our Intrusion Detection System.</p>
        </Link>

        <Link to="/predict" className="placard">
          <h2>Predict</h2>
          <p>Upload your data and let our IDS detect.</p>
        </Link>

        <Link to="/capture" className="placard">
          <h2>Capture</h2>
          <p>Capture real time data.</p>
        </Link>
      </div>
    </div>
  );
};

export default Home;