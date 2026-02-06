import { NavLink } from "react-router-dom";

function Navbar() {
  const linkClass = ({ isActive }) =>
    isActive ? "text-yellow-400 font-semibold" : "hover:text-yellow-400";

  return (
    <nav className="bg-gray-800 px-6 py-4 flex justify-between items-center">
      <h1 className="text-xl font-bold text-yellow-400">Gold & Silver Predictor</h1>

      <div className="space-x-6">
        <NavLink to="/" className={linkClass} end>
          Dashboard
        </NavLink>
        <NavLink to="/predict" className={linkClass}>
          Predict
        </NavLink>
        <NavLink to="/history" className={linkClass}>
          History
        </NavLink>
      </div>
    </nav>
  );
}

export default Navbar;
