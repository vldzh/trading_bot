"""
Polling-сервис: опрос платформы, инференс, отправка сигналов.
"""
import asyncio
import aiohttp
import logging
import pandas as pd
from aiohttp import BasicAuth
from config import PLATFORM_HOST, API_PASSWORD, API_USER, POLL_INTERVAL, LOOKBACK_STEPS, SOURCE_TAG
from inference import TradingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

AUTH = BasicAuth(API_USER, API_PASSWORD)


class SimpleMLService:
    def __init__(self, model: TradingModel):
        self.model = model
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def fetch_windows(self):
        """Получает готовые окна фич."""
        url = f"{PLATFORM_HOST}/api/ml/ds/feature-windows"
        try:
            async with self.session.get(url, auth=AUTH, params={"readyOnly": "true"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('items', [])
                else:
                    logger.error(f"Fetch error {resp.status}: {await resp.text()}")
                    return None
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return None
    
    async def send_signal(self, payload: dict):
        """Отправляет сигнал в платформу."""
        url = f"{PLATFORM_HOST}/api/signals/ingest"
        try:
            async with self.session.post(url, auth=AUTH, json=payload) as resp:
                if resp.status == 200:
                    logger.info(f"✅ Signal sent: {payload['symbol']} {payload['signal']}")
                    return True
                else:
                    logger.error(f"Send error {resp.status}: {await resp.text()}")
                    return False
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False
    
    async def process_window(self, window: dict):
        """Обрабатывает одно окно."""
        symbol = window.get('symbol', 'UNKNOWN')
        
        # Пропускаем WARMUP
        if window.get('state') != 'READY':
            logger.debug(f"[{symbol}] State={window.get('state')}, skip")
            return
        
        features = window.get('features')
        if not features or len(features) < LOOKBACK_STEPS:
            logger.warning(f"[{symbol}] Not enough features")
            return
        
        # Конвертируем в DataFrame
        columns = ['rd_value', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(features, columns=columns)
        
        # Инференс
        result = self.model.predict(df)
        logger.info(f"[{symbol}] Prediction: {result['signal']} (rating={result['rating']:.2f})")
        
        # Если HOLD — не отправляем
        if result['signal'] is None:
            return
        
        # Отправляем сигнал
        payload = {
            "symbol": symbol,
            "timestamp": window.get('windowEndTimestamp'),
            "signal": result['signal'],
            "price": float(df['close'].iloc[-1]),
            "rating": result['rating'],
            "source": SOURCE_TAG
        }
        await self.send_signal(payload)
    
    async def run_cycle(self):
        """Один цикл polling'а."""
        logger.info("🔄 Fetching windows...")
        windows = await self.fetch_windows()
        
        if windows is None:
            logger.error("❌ Failed to fetch windows")
            return
        
        logger.info(f"📦 Got {len(windows)} windows")
        
        for window in windows:
            try:
                await self.process_window(window)
            except Exception as e:
                logger.error(f"Error processing window: {e}")
    
    async def start(self):
        """Основной цикл."""
        logger.info(f"🚀 Starting ML Service | Host: {PLATFORM_HOST} | Interval: {POLL_INTERVAL}s")
        
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            
            logger.info(f"😴 Sleeping {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)
