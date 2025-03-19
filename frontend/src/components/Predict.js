import React, { useState } from "react";
import { predictCSV } from "../services/api";
import "./Predict.css"; // Import CSS for styling

const Predict = () => {
  const [file, setFile] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
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
        <input type="file" accept=".csv" onChange={handleFileChange} />
        {file && <p className="file-name">{file.name}</p>}
      </div>

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {error && <p className="error">{error}</p>}

      {downloadUrl && (
        <a href={downloadUrl} download="predictions.csv" className="download-link">
          Download Predictions
        </a>
      )}
    </div>
  );
};

export default Predict;
