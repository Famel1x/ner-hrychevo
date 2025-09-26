from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    app_name: str = "NER API"
    # где лежит модель (локальный путь) — можно переопределить через ENV
    model_path: str = os.getenv("MODEL_PATH", "artifacts/models/ner/best")
    # количество потоков под CPU-инференс
    max_workers: int = max(2, (os.cpu_count() or 2) * 2)
    # SLA < 1 секунды — ставим охранный таймаут
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", 0.95))

settings = Settings()