import React, { useState } from "react";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import "./Capture.css"; // Import CSS for styling
import axios from "axios"; // Import axios for making HTTP requests

const Capture = () => {
  const { logout } = useAuth0(); // Destructure logout from useAuth0
  const [duration, setDuration] = useState(""); // State for duration input
  const [csvData, setCsvData] = useState([]); // State for CSV data
  const [error, setError] = useState(""); // State for error messages
  const [loading, setLoading] = useState(false); // State for loading

  const handleCapture = async () => {
    try {
      setError(""); // Clear previous errors
      setLoading(true); // Set loading to true
      const response = await axios.post("http://localhost:8000/capture", {
        interface: "Wi-Fi", // You can change this to the desired interface
        duration: parseInt(duration), // Convert duration to an integer
      });

      // Assuming the response is a CSV string
      const csvText = response.data;
      const rows = csvText.split("\n").slice(0, 101); // Get the first 100 rows
      setCsvData(rows);
    } catch (err) {
      setError("Error capturing data. Please try again.");
      console.error(err);
    } finally {
      setLoading(false); // Set loading to false after the request completes
    }
  };

  const downloadCSV = () => {
    const csvContent = csvData.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "captured_data.csv"); // Set the file name
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="capture-container">
      <h1>Capture Data</h1>
      <input
        type="number"
        placeholder="Enter duration in seconds"
        value={duration}
        onChange={(e) => setDuration(e.target.value)}
      />
      <button className="capture-button" onClick={handleCapture} disabled={loading}>
        {loading ? "Loading..." : "Capture"}
      </button>

      {error && <p className="error">{error}</p>}

      {/* Download CSV Button positioned above the table */}
      {csvData.length > 0 && (
        <button className="download-button" onClick={downloadCSV}>
          Download CSV
        </button>
      )}

      {csvData.length > 0 && (
        <div className="table-container">
          <div className="scrollable-table">
            <table className="csv-table">
              <thead>
                <tr>
                  {csvData[0].split(",").map((header, index) => (
                    <th key={index}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvData.slice(1).map((row, index) => (
                  <tr key={index}>
                    {row.split(",").map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Logout Button */}
      <button className="logout-button" onClick={() => logout({ returnTo: window.location.origin })}>
        Logout
      </button>
    </div>
  );
};

export default Capture;