from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import io
from inference import TradingModel

app = FastAPI(title="ML Trading Zone Predictor")
model = TradingModel('model_weights.pkl')

@app.post("/predict")
    return {"prediction": "1"}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = f"""
    <html>
        <head>
            <title>ML Signal Zone Visualizer</title>
            <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        </head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>Тестирование модели: Загрузите файл</h2>
            <p style="color: gray; font-size: 14px;">🛠 <i>Дата и время генерации текущей модели: <b>{model.trained_at}</b></i></p>

            <input type="file" id="csvFile" accept=".csv" />
            <button onclick="uploadFile()" style="padding: 5px 15px; background: #007bff; color: white; border: none; cursor:pointer;">Отправить</button>
            <h3 id="resultText"></h3>
            <div id="chart"></div>

            <script>
                async function uploadFile() {{
                    const file = document.getElementById('csvFile').files[0];
                    if (!file) return alert("Выберите файл!");

                    const formData = new FormData();
                    formData.append("file", file);

                    document.getElementById('resultText').innerText = "Загрузка и просчет...";
                    const response = await fetch('/upload_csv', {{ method: "POST", body: formData }});

                    if (!response.ok) {{
                        const err = await response.json();
                        document.getElementById('resultText').innerText = "Ошибка: " + err.detail;
                        return;
                    }}

                    const result = await response.json();
                    document.getElementById('resultText').innerText = "Последняя спрогнозированная зона: " + (result.final_action === 1 ? "BUY (1)" : "SELL (-1)");

                    const trace1 = {{ x: result.times, y: result.close, type: 'scatter', name: 'Price', line: {{color: 'black'}} }};
                    const trace2 = {{ x: result.times, y: result.rd_values, type: 'scatter', name: 'RD Value', line: {{color: 'blue'}}, yaxis: 'y2', fill: 'tozeroy' }};

                    const layout = {{ 
                        title: 'Predicted Zones, Price, and RD Value', 
                        height: 700, 
                        yaxis: {{ domain: [0.3, 1], title: 'Price' }}, 
                        yaxis2: {{ domain: [0, 0.2], title: 'RD Value' }}, 
                        shapes: [] 
                    }};

                    let start_t = result.times[0];
                    let current_sig = result.predictions[0];

                    for(let i=1; i<result.predictions.length; i++){{
                        if(result.predictions[i] !== current_sig || i === result.predictions.length - 1){{
                            layout.shapes.push({{
                                type: 'rect', xref: 'x', yref: 'paper',
                                x0: start_t, x1: result.times[i], y0: 0, y1: 1,
                                fillcolor: current_sig === 1 ? 'green' : 'red',
                                opacity: 0.25, line: {{width: 0}}
                            }});
                            start_t = result.times[i];
                            current_sig = result.predictions[i];
                        }}
                    }}

                    Plotly.newPlot('chart', [trace1, trace2], layout);
                }}
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload_csv")
async def process_csv(file: UploadFile = File(...)):
    content = await file.read()

    try:
        decoded_content = content.decode('utf-8-sig')
        first_line = decoded_content.split('\n')[0].strip()
        separator = ';' if ';' in first_line else ','
        
        # Read the CSV entirely without assuming headers yet
        df = pd.read_csv(io.StringIO(decoded_content), sep=separator, header=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат CSV: {str(e)}")

    # 1. Smart Header Detection
    first_row_vals = df.iloc[0].astype(str).str.lower().str.strip().tolist()
    
    if 'timestamp' in first_row_vals or 'close' in first_row_vals:
        # It HAS headers. Assign them and remove the first row.
        df.columns = first_row_vals
        df = df[1:].reset_index(drop=True)
    else:
        # It does NOT have headers. Enforce the 8-value rule.
        if len(df.columns) != 8:
            raise HTTPException(
                status_code=400, 
                detail=f"Данные без заголовков должны содержать ровно 8 значений в строке. Найдено столбцов: {len(df.columns)}"
            )
        # Apply the exact 8 column names you provided
        df.columns = ['timestamp', 'symbol', 'rd_value', 'open', 'high', 'low', 'close', 'volume']

    # 2. Force data types to numeric
    try:
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['rd_value'] = pd.to_numeric(df['rd_value'])
        df['close'] = pd.to_numeric(df['close'])
    except Exception:
        raise HTTPException(status_code=400, detail="Ошибка типов: Невозможно преобразовать timestamp, rd_value или close в числа.")

    # 3. Trim unused columns
    required_cols = ['timestamp', 'rd_value', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"В CSV нет обязательных столбцов: {missing_cols}")

    df = df[required_cols].copy()
    
    # 4. Convert timestamp to readable time for the chart
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    times = df['timestamp'].dt.strftime('%H:%M:%S').tolist()
    close_prices = df['close'].tolist()
    rd_values = df['rd_value'].tolist()

    predictions = []
    for i in range(len(df)):
        if i < model.M:
            predictions.append(1)
        else:
            window_df = df.iloc[i - model.M : i + 1]
            pred = model.predict(window_df)
            
            if isinstance(pred, (list, pd.Series, np.ndarray)):
                 pred = int(pred[-1] if len(pred) > 0 else 1) 
            else:
                 pred = int(pred)
                 
            predictions.append(pred)

    return {
        "times": times, 
        "close": close_prices, 
        "rd_values": rd_values, 
        "predictions": predictions, 
        "final_action": predictions[-1]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
