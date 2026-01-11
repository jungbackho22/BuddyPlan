from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
Likert = Literal[1, 2, 3, 4, 5, 6]

@dataclass
class FactorScore:
    name: str
    mean: float
    percent: int
    level: Literal["낮음", "중간", "높음"]

def mean_to_percent(m: float) -> int:
    # 1 -> 0, 6 -> 100
    return round(((m - 1.0) / 5.0) * 100)

def level_from_mean(m: float) -> Literal["낮음", "중간", "높음"]:
    if m <= 2.5:
        return "낮음"
    if m <= 4.0:
        return "중간"
    return "높음"

def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

def score_factors(responses: Dict[str, int]) -> List[FactorScore]:
    factors = load_json("factors.json")["factors"]
    out: List[FactorScore] = []
    for f in factors:
        vals = [int(responses[i]) for i in f["item_ids"] if i in responses]
        m = round(sum(vals) / len(vals), 2) if vals else 0.0
        out.append(FactorScore(
            name=f["name"],
            mean=m,
            percent=mean_to_percent(m) if vals else 0,
            level=level_from_mean(m) if vals else "낮음",
        ))
    out.sort(key=lambda x: x.mean, reverse=True)  # 점수 높을수록 어려움 큼
    return out

FACTOR_TO_CODES: Dict[str, List[str]] = {
    "사회인지·맥락이해": ["Atb", "Que", "Idm"],
    "상호작용 기술·비언어": ["Nvr", "Que", "Apr"],
    "사회적 동기·회피/불안": ["Wdr", "Apr"],
    "반복·집착/감각·경직": ["Rbh", "Rin"],
    "자기표현·의사소통 유연성": ["Ver"],
    "정서·자기조절": ["Nvr", "Rbh"],
}

def recommend_scenarios(top_factor_names: List[str], max_n: int = 3) -> List[str]:
    scenarios = load_json("scenarios.json")["scenarios"]

    codes = set()
    for n in top_factor_names[:2]:
        codes.update(FACTOR_TO_CODES.get(n, []))

    pool = [s for s in scenarios if (s.get("code") or "").strip() in codes]

    picked: List[str] = []
    seen_tag = set()

    for s in pool:
        tag = (s.get("tags") or [""])[0]
        if tag not in seen_tag:
            picked.append(s["id"])
            if tag:
                seen_tag.add(tag)
        if len(picked) >= max_n:
            return picked

    for s in pool:
        if s["id"] not in picked:
            picked.append(s["id"])
        if len(picked) >= max_n:
            return picked

    for s in scenarios:
        if s["id"] not in picked:
            picked.append(s["id"])
        if len(picked) >= max_n:
            return picked

    return picked
