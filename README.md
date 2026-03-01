# ML Trading Zone Predictor

A FastAPI-based microservice for predicting financial trading zones (BUY/SELL) using a custom Machine Learning model. 

---

##  Main API Endpoint: `POST /predict`

The primary endpoint designed for live trading execution bots. It evaluates a historical time-series sequence (N steps) and returns the predicted trading zone for the latest data point.

**Request Format (Strictly Headerless Arrays):**
The payload must be a JSON object containing a `features` array. This array must contain multiple sub-arrays representing historical data over several days/periods. 

Each sub-array **must** contain exactly 8 values in this specific order:
`[timestamp, symbol, rd_value, open, high, low, close, volume]`

**Example Request:**
{
  "features": [
    [1769875320000, "ARC", -0.010200000, 0.15000, 0.15100, 0.14900, 0.15050, 5000],
    [1769961720000, "ARC", -0.011600572, 0.15358, 0.15392, 0.15283, 0.15285, 7258]
  ]
}

**Example Response:**
Returns the predicted action based on the sequence: `1` (Buy), `-1` (Sell), or `0` (Neutral).

{
  "prediction": 1
}

---

##  Web UI & Visualizer: `GET /`

Navigate to `http://<YOUR_IP>:8000/` in your browser to access the interactive web interface. 

This endpoint is dedicated to human testing and visual backtesting. You can manually upload CSV files (with or without headers, but containing the same 8 columns) and visualize the predicted trading zones overlaid on price and `rd_value` Plotly charts.

---

## Installation & Setup

1. **Clone the repository**:
    git clone git@github.com:YOUR_USERNAME/trading_bot.git
    cd trading_bot

2. **Create and activate a virtual environment**:
    python3 -m venv venv
    source venv/bin/activate

3. **Install dependencies**:
    pip install -r requirements.txt

## Running the Server

Start the application using Uvicorn. Binding to `0.0.0.0` allows external access to the server.

    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Auto-generated interactive API documentation (Swagger) is available at `http://<YOUR_IP>:8000/docs`.
