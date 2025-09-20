from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

VALID_KINDS = {"TYPE", "BRAND", "VOLUME", "PERCENT"}

def _bio_to_token_spans(labels: List[str]) -> List[Tuple[int,int,str]]:
    """
    BIO -> список (start_tok, end_tok_exclusive, kind).
    Робастно: I без B начинает новую сущность.
    """
    spans: List[Tuple[int,int,str]] = []
    start: Optional[int] = None
    cur: Optional[str] = None
    for i, lab in enumerate(labels):
        if not lab or lab == "O":
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
            continue
        if "-" in lab:
            pref, k = lab.split("-", 1)
        else:
            pref, k = "B", lab
        if k not in VALID_KINDS:
            # неизвестные теги игнорируем как O
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
            continue
        if pref == "B":
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = i, k
        elif pref == "I":
            if start is None or cur != k:
                if start is not None and cur is not None:
                    spans.append((start, i, cur))
                start, cur = i, k
        else:
            # неожиданный префикс -> закрываем
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
    if start is not None and cur is not None:
        spans.append((start, len(labels), cur))
    return spans

def _first_valid_offset(offsets: List[Tuple[int,int]], i: int, j: int) -> Optional[Tuple[int,int]]:
    """найти первый токен с непустым оффсетом в [i, j)."""
    for k in range(i, j):
        s, e = offsets[k]
        if s != e:
            return s, e
    return None

def _last_valid_offset(offsets: List[Tuple[int,int]], i: int, j: int) -> Optional[Tuple[int,int]]:
    """найти последний токен с непустым оффсетом в [i, j)."""
    for k in range(j-1, i-1, -1):
        s, e = offsets[k]
        if s != e:
            return s, e
    return None

def _trim_spaces(text: str, s: int, e: int) -> Tuple[int,int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e-1].isspace():
        e -= 1
    return s, e

def _merge_same_type_char_spans(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """
    Мерж пересекающихся/смежных спанов одного и того же типа.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda x: (x[2], x[0], x[1]))  # по типу, затем по началу
    out: List[Tuple[int,int,str]] = []
    cs, ce, ck = spans[0]
    for s, e, k in spans[1:]:
        if k == ck and s <= ce:  # пересекаются/касаются
            ce = max(ce, e)
        else:
            out.append((cs, ce, ck))
            cs, ce, ck = s, e, k
    out.append((cs, ce, ck))
    return out

def _drop_overlaps_by_longest(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """
    Убираем пересечения между разными типами.
    Жадно берём более длинные, при равной длине — более ранние.
    """
    # сортируем по длине убыв., затем по началу
    order = sorted(spans, key=lambda x: (-(x[1]-x[0]), x[0], x[2]))
    kept: List[Tuple[int,int,str]] = []
    for s, e, k in order:
        overlap = False
        for (ks, ke, kk) in kept:
            if not (e <= ks or s >= ke):
                overlap = True
                break
        if not overlap:
            kept.append((s, e, k))
    kept.sort(key=lambda x: (x[0], x[1]))
    return kept

def bio_postprocess(
    labels: List[str],
    offsets: List[Tuple[int,int]],
    text: str,
    *,
    trim_whitespace: bool = True,
    merge_same_type: bool = True,
    remove_overlaps: bool = True,
    return_bio_prefix: bool = False,
) -> List[Dict[str, Any]]:
    """
    Главная функция постпроцесса.
    :param labels: BIO-метки на токенах
    :param offsets: список (start_char, end_char) на токен
    :param text: исходный текст (для тримминга пробелов)
    :param return_bio_prefix: если True — entity будет 'B-XXX', иначе просто 'XXX'
    :return: список спанов [{"start_index","end_index","entity"}, ...]
    """
    if len(labels) != len(offsets):
        raise ValueError(f"len(labels) != len(offsets): {len(labels)} vs {len(offsets)}")

    # 1) BIO -> токенные спаны
    tok_spans = _bio_to_token_spans(labels)  # (ti, tj, kind)

    # 2) токенные -> символьные
    char_spans: List[Tuple[int,int,str]] = []
    L = len(text)
    for ti, tj, kind in tok_spans:
        head = _first_valid_offset(offsets, ti, tj)
        tail = _last_valid_offset(offsets, ti, tj)
        if head is None or tail is None:
            continue
        s = max(0, min(head[0], L))
        e = max(0, min(tail[1], L))
        if trim_whitespace:
            s, e = _trim_spaces(text, s, e)
        if s < e:
            char_spans.append((s, e, kind))

    # 3) мержим одинаковые типы (на случай разрывов внутри одного вида)
    if merge_same_type:
        char_spans = _merge_same_type_char_spans(char_spans)

    # 4) убираем пересечения между типами
    if remove_overlaps:
        char_spans = _drop_overlaps_by_longest(char_spans)

    # 5) сбор ответа
    out: List[Dict[str, Any]] = []
    for s, e, k in char_spans:
        ent = f"B-{k}" if return_bio_prefix else k
        out.append({"start_index": int(s), "end_index": int(e), "entity": ent})
    return out

if __name__ == "__main__":
    text = "abtoys игрушки 2 л"
    tokens = ["ab", "##to", "##ys", "игрушки", "2", "л"]
    offsets = [(0,2),(2,4),(4,6),(7,14),(15,16),(17,18)]
    labels  = ["B-BRAND","I-BRAND","I-BRAND","B-TYPE","B-VOLUME","I-VOLUME"]
    spans = bio_postprocess(labels, offsets, text)
    # [{'start_index': 0, 'end_index': 6, 'entity': 'BRAND'},
    #  {'start_index': 7, 'end_index': 14, 'entity': 'TYPE'},
    #  {'start_index': 15, 'end_index': 18, 'entity': 'VOLUME'}]
    print(spans)
