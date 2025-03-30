import React, { useState } from "react";
import { generateCSV } from "../services/api";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import Papa from "papaparse"; // Import PapaParse for CSV parsing
import "./Generate.css"; // Import CSS for styling

const Generate = () => {
  const { logout } = useAuth0(); // Destructure logout from useAuth0
  const [numSamples, setNumSamples] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [csvData, setCsvData] = useState([]); // State to hold parsed CSV data
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

      // Read and parse the generated CSV file for preview
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

      {/* Display the CSV data in a table format */}
      {csvData.length > 0 && (
        <div className="file-preview">
          <h3>Generated CSV Preview:</h3>
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

      {/* Logout Button */}
      <button className="logout-button" onClick={() => logout({ returnTo: window.location.origin })}>
        Logout
      </button>
    </div>
  );
};

export default Generate;