#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_bio.py — подготовка BIO-разметки из train.csv
- Читает CSV с колонками: sample, annotation
- Санитизирует спаны (клип границ, выкидывает 'O'/'0')
- Токенизирует (по умолчанию HF AutoTokenizer ruBERT; есть фолбэк на regex)
- Строит BIO по токенам с сохранением offset_mapping (символьные индексы)
- Сохраняет JSONL (tokens, offsets, labels) и CoNLL

Пример:
python prep_bio.py --in /path/train.csv --out out_dir \
  --tokenizer hf --hf-name ai-forever/ruBert-base
"""
import utils.bio_prep 

import argparse
import ast
import io
import json
import os
import re
import sys
from typing import List, Tuple, Dict

import pandas as pd


# ====== Константы и regex ======
VALID_KINDS = {"TYPE", "BRAND", "VOLUME", "PERCENT"}
WORD_RE = re.compile(r"\d+[.,]?\d*|[A-Za-zА-Яа-яЁё%]+|[^\sA-Za-zА-Яа-яЁё0-9]", re.UNICODE)


# ====== Утилиты логов ======
def log(msg: str):
    print(f"[prep_bio] {msg}", flush=True)


# ====== Чтение и санитайз аннотаций ======
def parse_ann_cell(cell) -> List[Tuple[int, int, str]]:
    """annotation: строка со списком кортежей (start, end, tag)"""
    try:
        ann = ast.literal_eval(cell) if isinstance(cell, str) else (cell or [])
        if not isinstance(ann, (list, tuple)):
            return []
        out = []
        for item in ann:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            s, e, t = item
            out.append((int(s), int(e), str(t)))
        return out
    except Exception:
        return []


def sanitize_ann(sample_text: str, anns: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    - Клип границ в [0, len(text)]
    - '0' -> 'O'
    - Удаляем 'O'-спаны (для BIO не нужны)
    """
    L = len(sample_text)
    cleaned = []
    for (s, e, t) in anns:
        t = 'O' if t == '0' else t
        s = max(0, min(int(s), L))
        e = max(0, min(int(e), L))
        if s >= e:
            continue
        if t == 'O':
            continue
        cleaned.append((s, e, t))
    return cleaned


# ====== Мерж символных спанов по виду сущности ======
def merge_char_spans(ann_list: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    На вход: (start,end,tag) где tag вида B-XXX или I-XXX (или просто XXX).
    Выход: слитые интервалы (start,end,kind) по сущности kind ∈ VALID_KINDS.
    """
    items = []
    for s, e, t in ann_list:
        kind = t.split("-", 1)[1] if "-" in t else t
        if kind not in VALID_KINDS:
            continue
        items.append((s, e, kind))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1]))
    merged = []
    cur_s, cur_e, cur_k = items[0]
    for s, e, k in items[1:]:
        if k == cur_k and s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_k))
            cur_s, cur_e, cur_k = s, e, k
    merged.append((cur_s, cur_e, cur_k))
    return merged


# ====== Токенизация ======
def tokenize_regex(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, offsets = [], []
    for m in WORD_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def build_hf_tokenizer(hf_name: str):
    """
    Возвращает (tokenizer, is_fast)
    Если нет transformers или fast-токенайзера — падаем на regex.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        # Проверим что fast и даёт offset_mapping
        test = tok("тест", return_offsets_mapping=True, add_special_tokens=False)
        _ = test.get("offset_mapping", None)
        if _ is None:
            log("HF токенайзер не возвращает offset_mapping → фолбэк на regex")
            return None, False
        return tok, True
    except Exception as e:
        log(f"HF токенайзер недоступен ({e}) → фолбэк на regex")
        return None, False


def tokenize_hf(tok, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets = [(int(s), int(e)) for (s, e) in enc["offset_mapping"]]
    # Отфильтруем пустые оффсеты (на всякий)
    toks2, offs2 = [], []
    for t, (s, e) in zip(tokens, offsets):
        if s == e:
            continue
        toks2.append(t)
        offs2.append((s, e))
    return toks2, offs2


# ====== BIO разметка по токенам ======
def spans_to_bio_for_tokens(
    offsets: List[Tuple[int, int]],
    spans_merged: List[Tuple[int, int, str]]
) -> List[str]:
    """
    Упрощённо: для каждого спана помечаем первый пересекающийся токен как B-*, остальные I-*.
    Если токен пересекается с несколькими спанами (редко) — оставляем первый по порядку.
    """
    labels = ["O"] * len(offsets)
    for s, e, kind in spans_merged:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if te <= s or ts >= e:
                continue
            # токен пересекает спан
            new_lab = ("B-" if first else "I-") + kind
            # если уже стоит метка, не переопределяем (разрешаем только первый спан)
            if labels[i] == "O":
                labels[i] = new_lab
            first = False
    return labels


# ====== Основной пайплайн ======
def process(
    in_csv: str,
    out_dir: str,
    text_col: str = "sample",
    ann_col: str = "annotation",
    sep: str = ";",
    tokenizer_kind: str = "hf",
    hf_name: str = "ai-forever/ruBert-base",
    save_conll: bool = True,
    save_jsonl: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Чтение датасета
    tried = []
    df = None
    for s in [sep, ";", ",", "\t"]:
        tried.append(s)
        try:
            df = pd.read_csv(in_csv, sep=s)
            if {text_col, ann_col}.issubset(df.columns):
                sep = s
                break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(
            f"Не удалось прочитать {in_csv}. Пробовал разделители: {tried}. "
            f"Ожидаю колонки: {text_col}, {ann_col}"
        )

    # 2) Санитайз аннотаций
    texts = df[text_col].astype(str).tolist()
    ann_raw = [parse_ann_cell(x) for x in df[ann_col].tolist()]
    ann_list = [sanitize_ann(t, a) for t, a in zip(texts, ann_raw)]

    # 3) Токенизатор
    hf_tok = None
    use_regex = tokenizer_kind != "hf"
    if tokenizer_kind == "hf":
        hf_tok, ok = build_hf_tokenizer(hf_name)
        use_regex = not ok
    log(f"Токенайзер: {'regex (fallback)' if use_regex else f'HF: {hf_name}'}")

    # 4) Конвертация во весь набор
    records = []
    bad_rows = 0
    for idx, (text, anns) in enumerate(zip(texts, ann_list)):
        spans_merged = merge_char_spans(anns)
        if use_regex:
            tokens, offsets = tokenize_regex(text)
        else:
            tokens, offsets = tokenize_hf(hf_tok, text)

        # пустые / странные кейсы — пустой набор токенов допустим
        labels = spans_to_bio_for_tokens(offsets, spans_merged)
        if len(labels) != len(tokens):
            bad_rows += 1

        rec = {
            "id": idx,
            "text": text,
            "tokens": tokens,
            "offsets": offsets,
            "labels": labels,
        }
        records.append(rec)

    # 5) Сохранение
    outputs = {}
    if save_jsonl:
        out_jsonl = os.path.join(out_dir, "train_bio.jsonl")
        with io.open(out_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        outputs["jsonl"] = out_jsonl

    if save_conll:
        out_conll = os.path.join(out_dir, "train_bio.conll")
        with io.open(out_conll, "w", encoding="utf-8") as f:
            for r in records:
                for tok, lab in zip(r["tokens"], r["labels"]):
                    f.write(f"{tok}\t{lab}\n")
                f.write("\n")
        outputs["conll"] = out_conll

    summary = {
        "total_rows": len(df),
        "converted_records": len(records),
        "bad_rows_label_mismatch": bad_rows,
        "tokenizer": "regex" if use_regex else f"hf:{hf_name}",
        "sep_used": sep,
        "outputs": outputs,
    }
    out_sum = os.path.join(out_dir, "bio_summary.json")
    with io.open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(json.dumps(summary, ensure_ascii=False))

    return summary


# ====== CLI ======
def main():
    p = argparse.ArgumentParser(description="Подготовка BIO-разметки из train.csv")
    p.add_argument("--in", dest="in_csv", required=True, help="Путь к train.csv")
    p.add_argument("--out", dest="out_dir", required=True, help="Каталог для сохранения")
    p.add_argument("--text-col", default="sample", help="Колонка с текстом (по умолчанию: sample)")
    p.add_argument("--ann-col", default="annotation", help="Колонка с аннотациями (по умолчанию: annotation)")
    p.add_argument("--sep", default=";", help="Разделитель CSV (по умолчанию ';', есть автоподбор)")
    p.add_argument("--tokenizer", choices=["hf", "regex"], default="hf", help="hf (по умолчанию) или regex")
    p.add_argument("--hf-name", default="ai-forever/ruBert-base", help="Имя/путь HF токенайзера")
    p.add_argument("--no-jsonl", action="store_true", help="Не сохранять JSONL")
    p.add_argument("--no-conll", action="store_true", help="Не сохранять CoNLL")

    args = p.parse_args()

    try:
        process(
            in_csv=args.in_csv,
            out_dir=args.out_dir,
            text_col=args.text_col,
            ann_col=args.ann_col,
            sep=args.sep,
            tokenizer_kind=args.tokenizer,
            hf_name=args.hf_name,
            save_conll=not args.no_conll,
            save_jsonl=not args.no_jsonl,
        )
    except Exception as e:
        log(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


