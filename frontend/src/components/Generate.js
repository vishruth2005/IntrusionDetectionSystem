import React, { useState } from "react";
import { generateCSV } from "../services/api";
import "./Generate.css"; // Import CSS for styling

const Generate = () => {
  const [numSamples, setNumSamples] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!numSamples || isNaN(numSamples) || numSamples <= 0) {
      setError("Please enter a valid number of samples.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const blob = await generateCSV(parseInt(numSamples));
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      setError("Failed to generate CSV. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="generate-container">
      <h1>Generate Synthetic Data</h1>
      <p>Enter the number of samples you want to generate.</p>

      <div className="input-group">
        <input
          type="number"
          value={numSamples}
          onChange={(e) => setNumSamples(e.target.value)}
          placeholder="Enter number of samples"
          min="1"
        />
        <button onClick={handleGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {downloadUrl && (
        <a href={downloadUrl} download="synthetic_data.csv" className="download-link">
          Download CSV
        </a>
      )}
    </div>
  );
};

export default Generate;
