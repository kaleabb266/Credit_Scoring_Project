// file path: src/App.jsx

import React, { useState } from "react";

const App = () => {
  const [formData, setFormData] = useState({
    Amount: "",
    Value: "",
    FraudResult: "",
    TotalTransactionAmount: "",
    AverageTransactionAmount: "",
    TransactionCount: "",
    TransactionStdDev: "",
    Recency: "",
    Frequency: "",
    Monetary: "",
    RFMS_Score: "",
    RFMS_Cluster: "",
    Seasonality: "",
    WoE: ""
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      const result = await response.json();
      setPrediction(result.prediction);
      console.lot(prediction)
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="App">
      <h1>Credit Scoring Model Prediction</h1>
      
      <form onSubmit={handleSubmit}>
        <label>
          Amount:
          <input type="number" name="Amount" value={formData.Amount} onChange={handleChange} />
        </label>
        <label>
          Value:
          <input type="number" name="Value" value={formData.Value} onChange={handleChange} />
        </label>
        <label>
          FraudResult:
          <input type="number" name="FraudResult" value={formData.FraudResult} onChange={handleChange} />
        </label>
        <label>
          TotalTransactionAmount:
          <input type="number" name="TotalTransactionAmount" value={formData.TotalTransactionAmount} onChange={handleChange} />
        </label>
        <label>
          AverageTransactionAmount:
          <input type="number" name="AverageTransactionAmount" value={formData.AverageTransactionAmount} onChange={handleChange} />
        </label>
        <label>
          TransactionCount:
          <input type="number" name="TransactionCount" value={formData.TransactionCount} onChange={handleChange} />
        </label>
        <label>
          TransactionStdDev:
          <input type="number" name="TransactionStdDev" value={formData.TransactionStdDev} onChange={handleChange} />
        </label>
        <label>
          Recency:
          <input type="number" name="Recency" value={formData.Recency} onChange={handleChange} />
        </label>
        <label>
          Frequency:
          <input type="number" name="Frequency" value={formData.Frequency} onChange={handleChange} />
        </label>
        <label>
          Monetary:
          <input type="number" name="Monetary" value={formData.Monetary} onChange={handleChange} />
        </label>
        <label>
          RFMS_Score:
          <input type="number" name="RFMS_Score" value={formData.RFMS_Score} onChange={handleChange} />
        </label>
        <label>
          RFMS_Cluster:
          <input type="number" name="RFMS_Cluster" value={formData.RFMS_Cluster} onChange={handleChange} />
        </label>
        <label>
          Seasonality:
          <input type="number" name="Seasonality" value={formData.Seasonality} onChange={handleChange} />
        </label>
        <label>
          WoE:
          <input type="number" name="WoE" value={formData.WoE} onChange={handleChange} />
        </label>

        <button type="submit">Get Prediction</button>
      </form>

      {/* Display prediction result */}
      {prediction && (
        <div>
          <h2>Prediction Result: {prediction}</h2>
        </div>
      )}
    </div>
  );
};

export default App;
