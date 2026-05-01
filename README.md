# Gold Price Prediction 🏆

A premium web application that predicts gold prices using advanced Machine Learning models. This project combines historical gold price data with powerful prediction algorithms to forecast future gold prices with high accuracy.

---

## 📸 Screenshots

### Home Page
![Gold Price Prediction - Home Page](/images/screenshot.png)

### Prediction Results
![Gold Price Prediction - Results](/images/Screenshot2.png/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Gold Price Prediction** is a sophisticated analytics platform that leverages machine learning to predict gold prices based on historical data. The application provides both a user-friendly web interface and REST API endpoints for seamless integration.

The model is trained on historical gold price data (GLD - Gold ETF prices) and uses temporal features to make accurate predictions for both past dates and future years.

---

## ✨ Features

- **Dual Prediction Modes:**
  - Predict by specific date (YYYY-MM-DD format)
  - Predict by year (YYYY format)

- **Advanced ML Models:**
  - Random Forest Regressor for accurate price prediction
  - Linear Regression for long-term trend analysis
  - Intelligent blending for future predictions

- **Trend Analysis:**
  - Automatic trend direction detection (Bullish/Bearish)
  - Price trend visualization and forecasting

- **Interactive Web Interface:**
  - Elegant, responsive UI with dark theme
  - Real-time prediction results
  - Visual trend charts and graphs

- **REST API:**
  - JSON endpoints for programmatic access
  - Easy integration with other applications

- **Model Transparency:**
  - Display model accuracy and MAPE (Mean Absolute Percentage Error)
  - Historical vs predicted price visualization

---

## 🛠️ Technologies Used

| Category | Technology | Version |
|----------|-----------|---------|
| **Backend Framework** | Flask | 2.3.0 |
| **Data Processing** | Pandas | 3.0.1 |
| **Numerical Computing** | NumPy | 1.26.4 |
| **Machine Learning** | scikit-learn | 1.5.0 |
| **Visualization** | Matplotlib | 3.7.2 |
| **WSGI Server** | Gunicorn | 21.2.0 |
| **Frontend** | HTML5, CSS3 | Modern |

---

## 📁 Project Structure

```
Gold-Price-Prediction/
│
├── app.py                      # Main Flask application
├── Gold Price.csv              # Historical gold price data
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── model/                      # Model storage (optional)
│   └── [trained models]
│
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
│
├── static/                     # CSS and static files
│   └── style.css              # Styling
│
├── images/                     # Image assets
│   ├── statue.png             # Hero section image
│   ├── angel.png              # Forecast section image
│   └── bg.png                 # Background image
│
└── instance/                   # Flask instance folder
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/hemant0hack/Gold-Price-Prediction.git
cd Gold-Price-Prediction
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

---

## 💻 Usage

### Web Interface

1. **Open the Application:**
   - Navigate to `http://localhost:5000` in your browser

2. **Make a Prediction:**
   - Click "GET STARTED" button
   - Enter a year (e.g., `2025`) or date (e.g., `2025-06-15`)
   - Click "Prediction" button

3. **View Results:**
   - See predicted gold price in INR per gram
   - View trend direction (Bullish/Bearish)
   - Analyze price trend visualizations

### API Usage

#### Get Prediction via API

**Request:**
```bash
GET /api/predict/<date>
```

**Examples:**

```bash
# Predict for a specific year
curl http://localhost:5000/api/predict/2025

# Predict for a specific date
curl http://localhost:5000/api/predict/2025-06-15
```

**Response (Success):**
```json
{
  "success": true,
  "date": "2025",
  "predicted_price": 7542.35,
  "currency": "INR",
  "model_accuracy": 95.42,
  "model_mape": 4.58
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Invalid date format. Use YYYY or YYYY-MM-DD"
}
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|-----------|
| `/` | GET | Home page | - |
| `/predict` | POST | Web form prediction | `date` (form data) |
| `/api/predict/<date>` | GET | API prediction endpoint | `date` (URL parameter) |
| `/plot` | GET | Historical price plot | - |
| `/plot_predict/<date>` | GET | Trend prediction plot | `date` (URL parameter) |
| `/images/<filename>` | GET | Serve images | `filename` |

---

## 🧠 Model Details

### Random Forest Regressor

**Purpose:** Primary prediction model for gold price forecasting

**Specifications:**
- **Algorithm:** Random Forest with 100 estimators
- **Features Used:**
  - Year
  - Month
  - Day
  - Day of Year
  - Quarter
  - Year Weight (normalized temporal weight)
- **Train/Test Split:** 80/20
- **Random State:** 42 (for reproducibility)

**Performance:**
- High accuracy on historical data
- Captures seasonal and temporal patterns
- Robust to outliers

### Linear Regression (Trend Model)

**Purpose:** Long-term trend analysis for future predictions

**Specifications:**
- **Uses:** Yearly average gold prices
- **Application:** Blended with RF model for predictions beyond the historical data range
- **Weight:** Increases with years into the future (up to 90% for very distant predictions)

### Prediction Logic

1. **For Past/Present Dates:** Direct Random Forest prediction
2. **For Future Dates:** Intelligent blending:
   - RF Prediction × (1 - trend_weight)
   - Linear Trend × trend_weight
   - Trend weight increases gradually for future years

---

## 📊 Performance Metrics

The model is evaluated using:

- **R² Score:** Measures how well predictions fit the test data
- **MAPE (Mean Absolute Percentage Error):** Average prediction error percentage
- **Model Accuracy:** 100 - MAPE

These metrics are displayed on the web interface for transparency.

---

## 🎨 UI/UX Features

### Design Elements
- **Elegant dark theme** with golden accents
- **Hero section** with compelling call-to-action
- **Professional gradient backgrounds** and glass-morphism effects
- **Responsive layout** that works on all devices

### Interactive Components
- **Input form** for flexible date/year entry
- **Real-time results** with instant predictions
- **Trend indicators** showing Bullish/Bearish directions
- **Dynamic charts** visualizing price trends
- **Model metrics** displaying accuracy and MAPE

---

## 🔮 Future Enhancements

- [ ] Add more machine learning models (LSTM, Prophet, XGBoost)
- [ ] Implement ensemble methods for better accuracy
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Historical accuracy comparison charts
- [ ] Export predictions to CSV/PDF
- [ ] Database integration for persistent storage
- [ ] User authentication and saved predictions
- [ ] Real-time data integration with live gold prices
- [ ] Mobile app version
- [ ] Model retraining scheduler
- [ ] Advanced analytics dashboard

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Improving model accuracy
- Adding new features
- UI/UX improvements
- Documentation enhancements
- Bug fixes

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Hemant Rathore**
- GitHub: [@hemant0hack](https://github.com/hemant0hack)
- Repository: [Gold-Price-Prediction](https://github.com/hemant0hack/Gold-Price-Prediction)
- LinkedIn: [Hemant Rathore](https://www.linkedin.com/in/hemant0hack/)

---

## 🙏 Acknowledgments

- Historical gold price data from GLD (Gold ETF)
- scikit-learn for powerful ML tools
- Flask framework for web development
- Community feedback and contributions

---

## 📈 Data Source

The project uses historical gold price data stored in `Gold Price.csv` with features:
- **Date:** Historical date
- **GLD:** Gold price in INR per gram

Data is preprocessed to extract temporal features (year, month, day, etc.) for model training.

---

**Built with ❤️ hemant0hack | Last Updated: May 2026**
