import asyncio
import sys
from inference import TradingModel
from app import SimpleMLService
from config import MODEL_PATH

async def main():
    try:
        model = TradingModel(MODEL_PATH)
        async with SimpleMLService(model) as service:
            await service.start()
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
