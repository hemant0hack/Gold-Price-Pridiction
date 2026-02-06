import { useEffect, useState } from "react";
import PriceCard from "../components/PriceCard";
import Loading from "../components/Loading";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function Home() {
  const [gold, setGold] = useState(null);
  const [silver, setSilver] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Try fetching predictions for common tickers (Gold futures GC=F, Silver SI=F)
    const fetchPrices = async () => {
      setLoading(true);
      try {
        const [gRes, sRes] = await Promise.all([
          fetch(`${API}/predict/GC=F`),
          fetch(`${API}/predict/SI=F`),
        ]);

        if (gRes.ok) setGold(await gRes.json());
        if (sRes.ok) setSilver(await sRes.json());
      } catch (err) {
        // backend may not be running; ignore and keep static values
        // eslint-disable-next-line no-console
        console.warn("Could not fetch live prices:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPrices();
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Market Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {loading && <Loading />}

        <PriceCard
          title="Gold Price (per 10g)"
          price={gold ? gold.last_close : "62,500"}
          color="text-yellow-400"
        />

        <PriceCard
          title="Silver Price (per kg)"
          price={silver ? silver.last_close : "74,200"}
          color="text-gray-300"
        />
      </div>
    </div>
  );
}

export default Home;
