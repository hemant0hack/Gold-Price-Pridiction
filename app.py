import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from flask import send_file

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

# Global variables for models
model = None
trend_model = None
gold_data = None
min_year = None
max_year = None
model_accuracy = None
model_mape = None

def train_models():
    """Train the models when server starts"""
    global model, trend_model, gold_data, min_year, max_year, model_accuracy, model_mape
    
    try:
        # Load data
        gold_data = pd.read_csv("Gold Price.csv")
        
        # Convert Date to datetime
        gold_data['Date'] = pd.to_datetime(gold_data['Date'])
        
        # Sort by date
        gold_data = gold_data.sort_values('Date')
        
        # Create features
        gold_data['Year'] = gold_data['Date'].dt.year
        gold_data['Month'] = gold_data['Date'].dt.month
        gold_data['Day'] = gold_data['Date'].dt.day
        gold_data['DayOfYear'] = gold_data['Date'].dt.dayofyear
        gold_data['Quarter'] = gold_data['Date'].dt.quarter
        
        # Calculate yearly average prices for trend
        yearly_avg = gold_data.groupby('Year')['GLD'].mean().reset_index()
        
        # Train linear trend model
        trend_model = LinearRegression()
        trend_model.fit(yearly_avg[['Year']], yearly_avg['GLD'])
        
        # Add year weight
        max_year = gold_data['Year'].max()
        min_year = gold_data['Year'].min()
        gold_data['YearWeight'] = (gold_data['Year'] - min_year) / (max_year - min_year)
        
        # Features for training
        X = gold_data[['Year', 'Month', 'Day', 'DayOfYear', 'Quarter', 'YearWeight']]
        y = gold_data['GLD']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        model_r2 = model.score(X_test, y_test)
        model_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        model_accuracy = 100 - model_mape
        
        print("[OK] Models trained successfully!")
        print(f"[INFO] Data range: {min_year} to {max_year}")
        print(f"[INFO] Average gold price: ₹{y.mean():.2f}")
        print(f"[INFO] Test Accuracy: {model_accuracy:.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Error training models: {e}")

def predict_gold_price(date_input):
    """Predict gold price based on date input"""
    try:
        print(f"[INFO] Predicting for: {date_input}")
        
        # Parse input
        if len(str(date_input)) == 4 and str(date_input).isdigit():
            date_obj = datetime(int(date_input), 6, 15)
            print(f"[INFO] Parsed as year: {date_obj.year}")
        else:
            date_obj = pd.to_datetime(date_input)
            print(f"[INFO] Parsed as date: {date_obj}")
        
        # Calculate year weight
        if min_year is None or max_year is None:
            print("[ERROR] min_year or max_year is None")
            return None, None
        
        year_weight = (date_obj.year - min_year) / (max_year - min_year)
        
        # Create input data
        input_data = pd.DataFrame([[
            date_obj.year,
            date_obj.month,
            date_obj.day,
            date_obj.timetuple().tm_yday,
            (date_obj.month - 1) // 3 + 1,
            year_weight
        ]], columns=['Year', 'Month', 'Day', 'DayOfYear', 'Quarter', 'YearWeight'])
        
        print(f"[INFO] Input features: {input_data.values[0]}")
        
        # Get prediction
        if model is None:
            print("[ERROR] Model is not trained.")
            return None, None
        model_prediction = model.predict(input_data)[0]
        print(f"[INFO] Model prediction: ₹{model_prediction:.2f}")

        # Apply trend adjustment for future years
        if date_obj.year > max_year:
            if trend_model is None:
                print("[ERROR] Trend model is not trained.")
                return None, None
            
            years_ahead = date_obj.year - max_year
            tw = min(0.9, 0.3 + years_ahead * 0.08)
            trend_value = trend_model.predict([[date_obj.year]])[0]
            final_prediction = (model_prediction * (1 - tw)) + (trend_value * tw)
            print(f"[INFO] Future blend — trend weight: {tw:.2f}, trend value: ₹{trend_value:.2f}")
        else:
            final_prediction = model_prediction

        print(f"[OK] Final prediction: ₹{final_prediction:.2f}")
        return final_prediction, date_obj
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return None, None

# Train models when server starts
train_models()

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """Home page with form"""
    return render_template(
        'index.html',
        accuracy_score=model_accuracy,
        mape=model_mape,
        Date=None,
        lastpridiction=None,
        selected_date=None,
        predicted_price=None,
        predicted_trend=None,
        trend_direction='upward',
        trend_nature='stable',
        error=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get date from form
        date_input = request.form.get('date')

        if not date_input:
            return render_template(
                'index.html',
                error='Please enter a date',
                accuracy_score=model_accuracy,
                mape=model_mape,
                Date=None,
                lastpridiction=None,
                selected_date=None,
                predicted_price=None,
                predicted_trend=None,
                trend_direction='upward',
                trend_nature='stable'
            )

        prediction, date_obj = predict_gold_price(date_input)

        if prediction is not None and date_obj is not None:
            if len(str(date_input)) == 4 and str(date_input).isdigit():
                formatted_date = f"Year {date_obj.year}"
            else:
                formatted_date = date_obj.strftime('%B %d, %Y')

            current_price = gold_data['GLD'].iloc[-1] if gold_data is not None else prediction
            trend_direction = 'upward' if prediction >= current_price else 'downward'
            trend_nature = 'bullish' if trend_direction == 'upward' else 'bearish'

            return render_template(
                'index.html',
                Date=formatted_date,
                lastpridiction=f"{round(prediction, 2):.2f}",
                selected_date=date_input,
                predicted_price=f"{round(prediction, 2):.2f}",
                predicted_trend=trend_nature,
                trend_direction=trend_direction,
                trend_nature=trend_nature,
                accuracy_score=model_accuracy,
                mape=model_mape,
                error=None
            )

        return render_template(
            'index.html',
            error='Invalid date format. Use YYYY or YYYY-MM-DD',
            accuracy_score=model_accuracy,
            mape=model_mape,
            Date=None,
            lastpridiction=None,
            selected_date=date_input,
            predicted_price=None,
            predicted_trend=None,
            trend_direction='upward',
            trend_nature='stable'
        )
    except Exception as e:
        print(f"Route error: {e}")
        return render_template(
            'index.html',
            error=str(e),
            accuracy_score=model_accuracy,
            mape=model_mape,
            Date=None,
            lastpridiction=None,
            selected_date=date_input if 'date_input' in locals() else None,
            predicted_price=None,
            predicted_trend=None,
            trend_direction='upward',
            trend_nature='stable'
        )

@app.route('/api/predict/<date>', methods=['GET'])
def predict_api_get(date):
    """API endpoint - GET request"""
    prediction, date_obj = predict_gold_price(date)

    if prediction is not None and date_obj is not None:
        return jsonify({
            'success': True,
            'date': date,
            'predicted_price': round(prediction, 2),
            'currency': 'INR',
            'model_accuracy': round(model_accuracy, 4) if model_accuracy is not None else None,
            'model_mape': round(model_mape, 2) if model_mape is not None else None
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Invalid date format. Use YYYY or YYYY-MM-DD'
        }), 400

@app.route('/plot')
def plot_graph():
    if gold_data is None or model is None:
        return "Model not trained yet", 400
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot historical price
    ax.plot(gold_data['Date'], gold_data['GLD'], label='Historical GLD Price', color='#f5d76e')
    
    # Plot generated fit by model
    X_all = gold_data[['Year', 'Month', 'Day', 'DayOfYear', 'Quarter', 'YearWeight']]
    y_pred_all = model.predict(X_all)
    ax.plot(gold_data['Date'], y_pred_all, label='Random Forest Fit', color='#ffffff', alpha=0.5, linestyle='--')
    
    ax.set_title("Gold Price Model: Historical Prediction", color='white', pad=15)
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price (INR)", color='white')
    ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    # Dark theme styling
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.grid(color='gray', linestyle='--', alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

@app.route('/plot_predict/<date>')
def plot_predict(date):
    if gold_data is None or model is None:
        return "Model not trained yet", 400
    
    try:
        if len(str(date)) == 4 and str(date).isdigit():
            date_obj = datetime(int(date), 6, 15)
        else:
            date_obj = pd.to_datetime(date)
    except:
        date_obj = datetime.now()
    
    target_year = date_obj.year

    # ---- HISTORICAL: yearly average from dataset ----
    yearly_avg = gold_data.groupby('Year')['GLD'].mean()
    hist_years = list(yearly_avg.index)
    hist_prices = list(yearly_avg.values)

    # ---- PREDICTED: model prediction for each year min_year → target_year ----
    year_range_full = list(range(min_year, target_year + 1))
    pred_prices = []

    for yr in year_range_full:
        yw = (yr - min_year) / (max_year - min_year)
        
        if yr == target_year:
            d = date_obj
        else:
            d = datetime(yr, 6, 15)
            
        input_data = pd.DataFrame([[
            yr, d.month, d.day, d.timetuple().tm_yday, (d.month - 1) // 3 + 1, yw
        ]], columns=['Year', 'Month', 'Day', 'DayOfYear', 'Quarter', 'YearWeight'])
        
        pred = model.predict(input_data)[0]
        
        if yr > max_year and trend_model is not None:
            years_ahead = yr - max_year
            tw = min(0.9, 0.3 + years_ahead * 0.08)
            trend_val = trend_model.predict([[yr]])[0]
            pred = (pred * (1 - tw)) + (trend_val * tw)
            
        pred_prices.append(pred)

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_years, hist_prices, marker='o', color='#888888',
            linestyle='-', linewidth=2, markersize=5, label='Historical Avg')
    ax.plot(year_range_full, pred_prices, marker='o', color='#f5d76e',
            linestyle='-', linewidth=2, markersize=5, label='Model Prediction')

    # Highlight the target year
    ax.scatter([target_year], [pred_prices[-1]], color='white', edgecolor='#f5d76e',
               s=180, zorder=5, linewidth=2)
    ax.annotate(f"₹{pred_prices[-1]:,.2f}",
                (target_year, pred_prices[-1]),
                textcoords="offset points", xytext=(0, 16),
                ha='center', color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="#111111", ec="#f5d76e", lw=1.5))

    # Dashed vertical line at max_year (historical ends, forecast begins)
    if target_year > max_year:
        ax.axvline(x=max_year, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(max_year + 0.1, min(pred_prices) * 0.995, 'Forecast →',
                color='#aaa', fontsize=9, va='bottom')

    ax.set_title(f"Gold Price: Historical vs Predicted (Up to {target_year})", color='white', pad=20)
    ax.set_xlabel("Year", color='white')
    ax.set_ylabel("Price (INR)", color='white')
    ax.legend(facecolor='#1e1e1e', edgecolor='#f5d76e', labelcolor='white', fontsize=10)

    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.grid(color='gray', linestyle='--', alpha=0.3)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)