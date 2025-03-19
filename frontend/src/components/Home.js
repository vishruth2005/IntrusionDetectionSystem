import React from "react";
import { Link } from "react-router-dom";
import "./Home.css"; // Import CSS

const Home = () => {
  return (
    <div className="home-container">
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
          <p>Capture real time data. (Coming Soon)</p>
        </Link>
      </div>
    </div>
  );
};

export default Home;
