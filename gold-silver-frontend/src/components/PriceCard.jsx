function PriceCard({ title, price, color }) {
  const formatted = typeof price === 'number' ? price.toLocaleString() : price;

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className={`text-3xl font-bold mt-3 ${color}`}>
        ₹ {formatted}
      </p>
    </div>
  );
}

export default PriceCard;
