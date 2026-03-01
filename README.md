# ML Trading Zone Predictor

A FastAPI-based microservice for predicting financial trading zones (BUY/SELL) using a custom Machine Learning model. 

The core of this application is the REST API, designed to be queried by live trading execution bots. It also includes a web-based UI (Plotly) for visual backtesting via CSV uploads.

---

## 🔌 Main API Endpoint: `POST /predict`

The primary endpoint expects a JSON payload containing an array of **historical data points (a time-series sequence)**. Because the model relies on past data to predict the current zone, the `features` array must contain several days/periods of data, not just a single row.

**All 8 data points are strictly required** in both formats:
`timestamp`, `symbol`, `rd_value`, `open`, `high`, `low`, `close`, `volume`.

### Format A: Dictionary (With Headers)
Useful for standard REST integrations.

**Request:**
{
  "features": [
    {
      "timestamp": 1769875320000,
      "symbol": "ARC",
      "rd_value": -0.010200000,
      "open": 0.15000,
      "high": 0.15100,
      "low": 0.14900,
      "close": 0.15050,
      "volume": 5000
    },
    {
      "timestamp": 1769961720000,
      "symbol": "ARC",
      "rd_value": -0.011600572,
      "open": 0.15358,
      "high": 0.15392,
      "low": 0.15283,
      "close": 0.15285,
      "volume": 7258
    }
  ]
}

### Format B: Raw Array (Headerless)
Optimized for speed and lower bandwidth when parsing live exchange websocket streams. Values must be in the exact order listed above.

**Request:**
{
  "features": [
    [1769875320000, "ARC", -0.010200000, 0.15000, 0.15100, 0.14900, 0.15050, 5000],
    [1769961720000, "ARC", -0.011600572, 0.15358, 0.15392, 0.15283, 0.15285, 7258]
  ]
}

### Expected Response
The model evaluates the entire historical sequence provided and returns the predicted action (`1` for Long/Buy, `-1` for Short/Sell) for the *latest* data point, along with the model's training timestamp.

{
  "prediction": 1,
  "model_date": "2026-03-01 14:11:46"
}

---

## Web UI & Visualizer

You can also test models manually via the web interface. 
Navigate to `http://<YOUR_IP>:8000/` in your browser to upload CSV files (with or without headers, requiring the same 8 columns) and visualize the predicted trading zones overlaid on price and `rd_value` charts.

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

##  Running the Server

Start the application using Uvicorn. Binding to `0.0.0.0` allows external access to the server.

    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Auto-generated interactive API documentation (Swagger) is available at `http://<YOUR_IP>:8000/docs`.
