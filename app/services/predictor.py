import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

Span = Dict[str, int | str]

class NerModel:
    """
    Обёртка реальной NER-модели.
    Сейчас — быстрая правило-based заглушка, которая:
      - бьёт текст на токены по \S+
      - первому слову ставит B-TYPE, остальным I-TYPE
      - для слов после первого включает ОДИН предшествующий пробел в диапазон,
        чтобы совпасть с примером ответа:
          "сгущенное молоко" -> [0..8], [9..15]
    Замените метод predict на вызов вашей ML-модели, вернув те же поля.
    """
    _token_re = re.compile(r"\S+")

    def __init__(self, model_path: str):
        self.model_path = model_path
        # здесь можно загрузить веса/токенайзер и т.п.
        # self.pipeline = load_pipeline(model_path)

    def predict(self, text: str) -> List[Span]:
        spans: List[Span] = []
        if not text:
            return spans

        tokens = list(self._token_re.finditer(text))
        if not tokens:
            return spans

        for i, m in enumerate(tokens):
            start = m.start()
            # Если сразу слева пробел — включаем его, чтобы совпасть с эталонным форматом.
            if i > 0 and start > 0 and text[start - 1] == " ":
                start -= 1
            end = m.end() - 1  # inclusive
            tag = "B-TYPE" if i == 0 else "I-TYPE"
            spans.append({"start_index": start, "end_index": end, "entity": tag})

        return spans


class AsyncPredictor:
    """
    Неблокирующая обёртка:
    sync-инференс модели выполняется в ThreadPoolExecutor,
    чтобы event loop FastAPI оставался свободным.
    """
    def __init__(self, model: NerModel, max_workers: int):
        self.model = model
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    async def predict(self, text: str) -> List[Span]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self.model.predict, text)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)