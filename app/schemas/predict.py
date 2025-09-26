from typing import List, Literal
from pydantic import BaseModel

class PredictIn(BaseModel):
    input: str

class Span(BaseModel):
    start_index: int   # inclusive
    end_index: int     # inclusive
    entity: Literal["B-TYPE", "I-TYPE"]

PredictOut = List[Span]