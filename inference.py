import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class TradingModel:
    def __init__(self, model_path='model_weights.pkl'):
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data['features']
        self.M = data['M']
        # ЧИТАЕМ ДАТУ ГЕНЕРАЦИИ (если старая модель без даты - пишем 'Неизвестно')
        self.trained_at = data.get('trained_at', 'Неизвестно') 
        self.scaler = StandardScaler()

    def predict(self, features_df: pd.DataFrame) -> int:
        # Берем только нужные колонки, игнорируя лишние
        required_cols = ['rd_value']
        # Убедимся, что нужный столбец есть
        if not all(col in features_df.columns for col in required_cols):
            raise ValueError(f"Отсутствуют обязательные колонки: {required_cols}")
            
        df = features_df.copy().tail(self.M + 1)
        
        df['rd_value_scaled'] = self.scaler.fit_transform(df['rd_value'].values.reshape(-1, 1)).flatten()
        
        for i in range(1, self.M + 1):
            df[f"rd_scaled_lag_{i}"] = df['rd_value_scaled'].shift(i)
            
        df[f"rd_rolling_mean_{self.M}"] = df['rd_value_scaled'].rolling(self.M, min_periods=1).mean()
        df[f"rd_rolling_std_{self.M}"] = df['rd_value_scaled'].rolling(self.M, min_periods=1).std().fillna(0)
            
        latest_features = df.iloc[[-1]][self.feature_names]
        prediction = int(self.model.predict(latest_features)[0])
        
        return prediction
