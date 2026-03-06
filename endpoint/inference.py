"""
Модуль инференса: загрузка модели, генерация фич, прогноз.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TradingModel:
    """
    Ожидает на вход: DataFrame с 60 барами, колонки ['rd_value', 'open', 'high', 'low', 'close', 'volume']
    Возвращает: dict с сигналом и уверенностью
    """
    
    REQUIRED_COLUMNS = ['rd_value', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, model_path: str):
        """Загружает модель и метаданные из файла."""
        logger.info(f"Loading model from {model_path}")
        
        try:
            data = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.model = data['model']
        self.feature_names = data.get('features', [])
        self.M = data.get('M', 3)  # Размер окна для лагов
        self.trained_at = data.get('trained_at', 'unknown')
        
        # Scaler для нормализации
        self.scaler: Optional[StandardScaler] = data.get('scaler')
        if self.scaler is None:
            logger.warning("Scaler not found in model file. Using fallback normalization.")
        
        logger.info(f"Model loaded. Trained at: {self.trained_at}, M={self.M}")
    
    def _normalize_rd(self, rd_values: pd.Series) -> np.ndarray:
        """Нормализует rd_value с использованием scaler или fallback."""
        rd_array = rd_values.values.reshape(-1, 1)
        
        if self.scaler is not None:
            return self.scaler.transform(rd_array).flatten()
        else:
            # Fallback: robust scaling на окне
            median = np.median(rd_array)
            iqr = np.percentile(rd_array, 75) - np.percentile(rd_array, 25)
            return ((rd_array - median) / (iqr + 1e-8)).flatten()
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерирует фичи из сырых данных (лаги, rolling-статистики)."""
        df = df.copy()
        
        # Нормализация rd_value
        df['rd_value_scaled'] = self._normalize_rd(df['rd_value'])
        
        # Лаги
        for i in range(1, self.M + 1):
            df[f"rd_scaled_lag_{i}"] = df['rd_value_scaled'].shift(i)
        
        # Rolling-статистики
        df[f"rd_rolling_mean_{self.M}"] = df['rd_value_scaled'].rolling(self.M, min_periods=1).mean()
        df[f"rd_rolling_std_{self.M}"] = df['rd_value_scaled'].rolling(self.M, min_periods=1).std().fillna(0)
        
        return df
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, any]:
        """
        Выполняет прогноз.
        
        Args:
            features_df: DataFrame с колонками ['rd_value', 'open', 'high', 'low', 'close', 'volume']
                        Минимум 60 строк, порядок: oldest -> newest
        
        Returns:
            dict: {
                'signal': 'BUY' | 'SELL' | None,  # None = HOLD (не отправлять сигнал)
                'rating': float,                   # Уверенность модели [0.0, 1.0]
                'prediction_class': int,           # Внутренний класс: 1, -1, 0
                'reason': str | None               # Причина, если сигнал не сгенерирован
            }
        """
        # Валидация входных данных
        if len(features_df) < 60:
            return {
                'signal': None,
                'rating': 0.0,
                'prediction_class': 0,
                'reason': f'not_enough_bars: {len(features_df)} < 60'
            }
        
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in features_df.columns]
        if missing_cols:
            return {
                'signal': None,
                'rating': 0.0,
                'prediction_class': 0,
                'reason': f'missing_columns: {missing_cols}'
            }
        
        try:
            # Берём последние 60 баров
            df = features_df.tail(60).copy()
            
            # Генерация фич
            df_feat = self._generate_features(df)
            
            # Берём последнюю строку для прогноза
            latest = df_feat.iloc[[-1]]
            
            # Фильтруем только нужные фичи
            available_features = [f for f in self.feature_names if f in latest.columns]
            if not available_features:
                return {
                    'signal': None,
                    'rating': 0.0,
                    'prediction_class': 0,
                    'reason': 'no_matching_features'
                }
            
            X = latest[available_features]
            
            # Предсказание
            pred_class = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            
            # Маппинг классов в сигналы
            signal_map = {1: 'BUY', -1: 'SELL', 0: None}
            signal = signal_map.get(pred_class, None)
            
            # Rating = максимальная вероятность
            rating = float(max(proba))
            
            return {
                'signal': signal,
                'rating': round(rating, 4),
                'prediction_class': pred_class,
                'reason': None
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'signal': None,
                'rating': 0.0,
                'prediction_class': 0,
                'reason': f'prediction_error: {str(e)}'
            }
