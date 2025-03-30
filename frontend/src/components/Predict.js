import React, { useState } from "react";
import { predictCSV } from "../services/api";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import Papa from "papaparse"; // Import PapaParse for CSV parsing
import "./Predict.css"; // Import CSS for styling

const Predict = () => {
  const { logout } = useAuth0(); // Destructure logout from useAuth0
  const [file, setFile] = useState(null);
  const [csvData, setCsvData] = useState([]); // State to hold parsed CSV data
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);

    // Read and parse the CSV file
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      Papa.parse(text, {
        header: true, // Use the first row as header
        complete: (results) => {
          setCsvData(results.data.slice(0, 100)); // Limit to 100 rows
        },
      });
    };
    reader.readAsText(selectedFile); // Read the file as text
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const blob = await predictCSV(file);
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);

      // Read and parse the returned CSV file for preview
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        Papa.parse(text, {
          header: true, // Use the first row as header
          complete: (results) => {
            setCsvData(results.data.slice(0, 100)); // Limit to 100 rows
          },
        });
      };
      reader.readAsText(blob); // Read the blob as text for preview
    } catch (err) {
      setError("Failed to process file. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="predict-container">
      <h1>Predict with AI</h1>
      <p>Upload a CSV file and get predictions.</p>

      <div className="file-upload">
        <label className="file-upload-label">
          Choose a file
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>
        {file && <p className={`file-name visible`}>{file.name}</p>} {/* Add visible class */}
      </div>

      {/* Display the CSV data in a table format */}
      {csvData.length > 0 && (
        <div className="file-preview">
          <h3>File Preview:</h3>
          <table className="csv-table">
            <thead>
              <tr>
                {Object.keys(csvData[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {csvData.map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((cell, cellIndex) => (
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <button className="predict-button" onClick={handlePredict} disabled={loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {error && <p className="error">{error}</p>}

      {downloadUrl && (
        <a href={downloadUrl} download="predictions.csv" className="download-link">
          Download Predictions
        </a>
      )}

      {/* Logout Button */}
      <button className="logout-button" onClick={() => logout({ returnTo: window.location.origin })}>
        Logout
      </button>
    </div>
  );
};

export default Predict;