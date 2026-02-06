import { useState } from "react";
import PriceCard from "../components/PriceCard";
import Loading from "../components/Loading";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function Predict() {
  const [symbol, setSymbol] = useState("AAPL");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/predict/${encodeURIComponent(symbol)}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Price Prediction</h2>

      <div className="flex gap-2 mb-4">
        <input
          className="px-3 py-2 rounded bg-gray-800 text-gray-100"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Enter symbol (e.g. AAPL, GC=F)"
        />
        <button
          className="bg-yellow-400 text-gray-900 px-4 rounded font-semibold"
          onClick={fetchPrediction}
          disabled={loading}
        >
          {loading ? <Loading /> : "Predict"}
        </button>
      </div>

      {error && <div className="text-red-400 mb-4">{error}</div>}

      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PriceCard
            title={`Last Close (${result.symbol})`}
            price={result.last_close}
            color={result.trend === "UP" ? "text-green-400" : "text-red-400"}
          />

          <PriceCard
            title={`Predicted Price (${result.symbol})`}
            price={result.predicted_price}
            color={result.trend === "UP" ? "text-green-400" : "text-red-400"}
          />
        </div>
      )}
    </div>
  );
}

export default Predict;
