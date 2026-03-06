✅ Что сделано:
Архитектура изменена на Pull: сервис сам опрашивает API каждые 60 сек, получает готовые окна по 60 баров
Сигналы теперь в формате BUY/SELL (HOLD = ничего не отправляем)
train.py исправлен: scaler теперь сохраняется в model_weights.pkl, нормализация стабильная между обучением и инференсом
inference.py адаптирован под 60 баров, есть fail-safe если данных меньше
app.py упрощён под один пароль (AUTH-02 с двумя пользователями пока не реализован на сервере, оставил задел)
✅ Локальное тестирование:
Модель загружается без ошибок
Прогноз на случайных данных проходит: выдаёт сигнал + rating
❌ Что не удалось протестировать:
API ms.as2eng1n.beget.tech возвращает 502 Bad Gateway от nginx
Интеграционный тест не прошёл — сервер пока не отвечает
Неясно, нужен ли VPN для доступа к хосту
📋 Файлы готовы:
inference.py (модель)
app.py (polling-сервис)
config.py (настройки)
main.py (точка входа)
train.py (обучение со scaler)



Для локального тестирования:
python -c "    
import pandas as pd, numpy as np
from inference import TradingModel

# Загружаем модель
model = TradingModel('model_weights.pkl')
print('✅ Model loaded')

# Генерируем 60 тестовых баров
df = pd.DataFrame({
    'rd_value': np.random.randn(60) * 0.1,
    'open': np.random.randn(60) * 100 + 42000,
    'high': np.random.randn(60) * 100 + 42100,
    'low': np.random.randn(60) * 100 + 41900,
    'close': np.random.randn(60) * 100 + 42050,
    'volume': np.random.randn(60) * 200 + 1000
})

# Прогноз
result = model.predict(df)
print(f'✅ Prediction: {result}')
"