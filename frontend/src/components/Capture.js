import React from "react";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import "./Capture.css"; // Import CSS for styling

const Capture = () => {
  const { logout } = useAuth0(); // Destructure logout from useAuth0

  return (
    <div className="capture-container">
      <h1>Capture Data</h1>
      <p>This feature is coming soon. Stay tuned!</p>

      {/* Logout Button */}
      <button className="logout-button" onClick={() => logout({ returnTo: window.location.origin })}>
        Logout
      </button>
    </div>
  );
};

export default Capture;