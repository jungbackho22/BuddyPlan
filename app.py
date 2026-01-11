# app.py (전체 교체본)
# 기능:
# - questions.json 중복 문항ID(P11 등) 자동 유일화(렌더링 안정화)
# - factors.json은 그대로 유지(요인 간 중복 문항 허용)
# - 1문항씩(이전/다음/제출) 방식
# - 사용자 정보(이름/나이/성별) 입력 받아 DB(SQLite)에 함께 저장
# - 시나리오 추천 고도화: 요인-코드 + 태그(키워드) 점수화 + 다양성(태그 중복 최소화)
#
# 실행:
#   streamlit run app.py

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# 기존 로직(score_factors)은 그대로 사용 (factors.json 유지 + 문항 중복 요인 반영 가능)
from logic import score_factors, FACTOR_TO_CODES  # type: ignore

# -----------------------------
# 기본 설정/데이터 로드
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "app.db"


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


QUESTIONS_RAW = load_json("questions.json")["questions"]
SCALE = load_json("questions.json")["scale"]
SCENARIO_LIST = load_json("scenarios.json")["scenarios"]
SCENARIOS: Dict[str, dict] = {s["id"]: s for s in SCENARIO_LIST}


def dedupe_questions_by_id(questions: List[dict]) -> List[dict]:
    """
    questions.json에 같은 id(P11)가 중복으로 들어있어도
    화면에서는 1번만 렌더링되도록 유일화합니다.
    - factors.json은 그대로 둬도 됨(요인 간 중복 문항 허용)
    - 응답은 문항ID 기준으로 1개만 저장됨
    """
    seen = set()
    unique: List[dict] = []
    for q in questions:
        qid = str(q.get("id", "")).strip()
        text = str(q.get("text", "")).strip()
        if not qid:
            continue
        if qid in seen:
            continue
        seen.add(qid)
        unique.append({"id": qid, "text": text})
    return unique


QUESTIONS = dedupe_questions_by_id(QUESTIONS_RAW)
QUESTION_IDS = [q["id"] for q in QUESTIONS]


# -----------------------------
# DB 유틸 (스키마 생성 + 컬럼 자동 추가)
# -----------------------------
def connect_db() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db() -> None:
    con = connect_db()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS assessments (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          created_at TEXT NOT NULL,
          responses_json TEXT NOT NULL,
          factor_scores_json TEXT NOT NULL,
          top_factors_json TEXT NOT NULL,
          recommended_scenarios_json TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scenario_attempts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          created_at TEXT NOT NULL,
          scenario_id TEXT NOT NULL,
          chosen_label TEXT NOT NULL,
          is_correct INTEGER NOT NULL
        );
        """
    )
    con.commit()

    # assessments 테이블에 사용자 정보 컬럼이 없으면 자동으로 추가
    existing_cols = [r[1] for r in con.execute("PRAGMA table_info(assessments);").fetchall()]
    if "user_name" not in existing_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN user_name TEXT;")
    if "user_age" not in existing_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN user_age INTEGER;")
    if "user_gender" not in existing_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN user_gender TEXT;")

    con.commit()
    con.close()


def now_iso() -> str:
    # Streamlit Cloud에서도 안정적으로 동작하도록 단순 ISO 문자열
    # (UTC 기준)
    import datetime

    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def insert_assessment(
    user_id: str,
    user_name: str,
    user_age: int,
    user_gender: str,
    responses: Dict[str, int],
    factor_scores: List[Dict[str, Any]],
    top_factors: List[Dict[str, Any]],
    recommended_ids: List[str],
) -> int:
    con = connect_db()

    cols = [r[1] for r in con.execute("PRAGMA table_info(assessments);").fetchall()]
    has_profile = all(c in cols for c in ["user_name", "user_age", "user_gender"])

    created_at = now_iso()

    if has_profile:
        cur = con.execute(
            """
            INSERT INTO assessments
            (user_id, created_at, user_name, user_age, user_gender,
             responses_json, factor_scores_json, top_factors_json, recommended_scenarios_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                user_name,
                user_age,
                user_gender,
                json.dumps(responses, ensure_ascii=False),
                json.dumps(factor_scores, ensure_ascii=False),
                json.dumps(top_factors, ensure_ascii=False),
                json.dumps(recommended_ids, ensure_ascii=False),
            ),
        )
    else:
        # 혹시 스키마가 적용되지 않은 경우에도 동작하게(안전장치)
        cur = con.execute(
            """
            INSERT INTO assessments
            (user_id, created_at, responses_json, factor_scores_json, top_factors_json, recommended_scenarios_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                json.dumps(responses, ensure_ascii=False),
                json.dumps(factor_scores, ensure_ascii=False),
                json.dumps(top_factors, ensure_ascii=False),
                json.dumps(recommended_ids, ensure_ascii=False),
            ),
        )

    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid


def latest_assessment(user_id: str) -> Optional[Dict[str, Any]]:
    con = connect_db()
    row = con.execute(
        """
        SELECT id, created_at, user_name, user_age, user_gender,
               factor_scores_json, top_factors_json, recommended_scenarios_json
        FROM assessments
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    con.close()

    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "user_name": row[2],
        "user_age": row[3],
        "user_gender": row[4],
        "factor_scores": json.loads(row[5]),
        "top_factors": json.loads(row[6]),
        "recommended_ids": json.loads(row[7]),
    }


def list_assessments(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    con = connect_db()
    rows = con.execute(
        """
        SELECT id, created_at, user_name, user_age, user_gender, top_factors_json
        FROM assessments
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    con.close()

    out = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "created_at": r[1],
                "user_name": r[2],
                "user_age": r[3],
                "user_gender": r[4],
                "top_factors": json.loads(r[5]),
            }
        )
    return out


def insert_attempt(user_id: str, scenario_id: str, chosen_label: str, is_correct: bool) -> None:
    con = connect_db()
    con.execute(
        """
        INSERT INTO scenario_attempts (user_id, created_at, scenario_id, chosen_label, is_correct)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, now_iso(), scenario_id, chosen_label, 1 if is_correct else 0),
    )
    con.commit()
    con.close()


# -----------------------------
# 추천 고도화(요인-코드 + 태그 키워드 + 다양성)
# -----------------------------
# 요인별로 “강화하고 싶은 태그/키워드” 운영 규칙(필요 시 여기만 수정)
FACTOR_TO_TAG_KEYWORDS: Dict[str, List[str]] = {
    "사회인지·맥락이해": ["눈치", "의도", "뉘앙스", "농담", "표정", "목소리"],
    "상호작용 기술·비언어": ["눈맞춤", "표정", "말투", "거리", "자세", "반응"],
    "사회적 동기·회피/불안": ["모임", "친구", "학교", "대화", "참여", "어울림"],
    "반복·집착/감각·경직": ["변화", "계획", "일정", "예민", "습관", "집착"],
    "자기표현·의사소통 유연성": ["부탁", "설명", "의견", "질문", "대답", "대화"],
    "정서·자기조절": ["불안", "긴장", "감정", "조절", "짜증", "진정"],
}


def score_scenario_for_user(
    scenario: dict,
    top_factor_names: List[str],
) -> int:
    """
    시나리오 추천 점수:
    - 코드 매칭: +50
    - 태그/텍스트 키워드 매칭: +10씩 (최대 30)
    """
    s_code = str(scenario.get("code", "")).strip()
    s_tags = scenario.get("tags") or []
    s_title = str(scenario.get("title", "")).strip()
    s_detail = str(scenario.get("detail", "")).strip()
    s_text_blob = " ".join([s_title, s_detail, " ".join(map(str, s_tags))])

    # 대상 코드 집합(상위 1~2개 요인 기준)
    target_codes = set()
    for fn in top_factor_names[:2]:
        for c in FACTOR_TO_CODES.get(fn, []):
            target_codes.add(c)

    score = 0

    # 1) 코드 매칭 점수
    if s_code and s_code in target_codes:
        score += 50

    # 2) 태그/텍스트 키워드 매칭 점수
    kw_hits = 0
    for fn in top_factor_names[:2]:
        for kw in FACTOR_TO_TAG_KEYWORDS.get(fn, []):
            if kw and kw in s_text_blob:
                kw_hits += 1
    score += min(kw_hits, 3) * 10  # 최대 30점

    return score


def pick_with_diversity(scored: List[Tuple[int, dict]], k: int = 3) -> List[str]:
    """
    점수 높은 순으로 뽑되,
    - 이미 선택된 시나리오들과 "첫 태그"가 같으면 우선순위를 낮추는 방식으로 다양성 확보
    """
    picked: List[dict] = []
    picked_ids: List[str] = []
    used_primary_tags = set()

    # 1차: 첫 태그 다양성 우선
    for s_score, s in scored:
        primary = (s.get("tags") or [""])[0]
        if primary and primary in used_primary_tags:
            continue
        picked.append(s)
        picked_ids.append(s["id"])
        if primary:
            used_primary_tags.add(primary)
        if len(picked_ids) >= k:
            return picked_ids

    # 2차: 점수 순으로 남은 것 채우기
    for s_score, s in scored:
        if s["id"] in picked_ids:
            continue
        picked_ids.append(s["id"])
        if len(picked_ids) >= k:
            return picked_ids

    return picked_ids


def recommend_scenarios_advanced(top_factor_names: List[str], k: int = 3) -> List[str]:
    scored: List[Tuple[int, dict]] = []
    for s in SCENARIO_LIST:
        scored.append((score_scenario_for_user(s, top_factor_names), s))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 점수가 모두 0일 수 있으니(키워드 미매칭) 그래도 상위 k는 뽑음
    return pick_with_diversity(scored, k=k)


# -----------------------------
# UI 유틸
# -----------------------------
def level_comment(level: str) -> str:
    if level == "낮음":
        return "현재 이 영역에서의 어려움이 비교적 적게 보고되었습니다."
    if level == "중간":
        return "상황에 따라 어려움이 나타날 수 있어, 시나리오 연습을 권장합니다."
    return "이 영역에서의 어려움이 비교적 크게 보고되었습니다. 추천 시나리오로 단계적으로 연습해보세요."


def validate_profile(name: str, age: int, gender: str) -> Tuple[bool, str]:
    if not name.strip():
        return False, "이름을 입력해주세요."
    if age < 1 or age > 120:
        return False, "나이는 1~120 사이로 입력해주세요."
    if gender not in ["남", "여", "기타", "선택안함"]:
        return False, "성별을 선택해주세요."
    return True, ""


def validate_all_answered(responses: Dict[str, int]) -> Tuple[bool, List[str]]:
    missing = [qid for qid in QUESTION_IDS if qid not in responses]
    return (len(missing) == 0, missing)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="버디플랜", layout="centered")
init_db()

# 사용자(간단): 세션별 UUID (로그인 없이 테스트용)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
USER_ID = st.session_state.user_id

# 1문항 진행 상태
if "step" not in st.session_state:
    st.session_state.step = 0

# 응답 저장
if "responses" not in st.session_state:
    st.session_state.responses = {}  # { "P01": 1~6, ... }

# 사용자 프로필
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "age": 10, "gender": "선택안함"}

st.title("버디플랜")
st.caption("6가지 주요 사회성 요인을 분석하고,아이에게 맞는 사회 훈련을 추천해 드립니다.")

tab1, tab2, tab3 = st.tabs(["검사(1문항씩)", "결과/추천", "내 기록"])


# -----------------------------
# 탭 1: 검사(1문항씩)
# -----------------------------
with tab1:
    st.subheader("1) 사용자 정보")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("이름", value=st.session_state.profile["name"])
    with c2:
        age = st.number_input("나이", min_value=1, max_value=120, value=int(st.session_state.profile["age"]))
    with c3:
        gender = st.selectbox("성별", options=["선택안함", "남", "여", "기타"], index=["선택안함", "남", "여", "기타"].index(st.session_state.profile["gender"]))

    st.session_state.profile = {"name": name, "age": int(age), "gender": gender}

    ok_profile, profile_msg = validate_profile(name, int(age), gender)
    if not ok_profile:
        st.info(profile_msg)

    st.divider()
    st.subheader("2) 문항 응답")
    st.write("점수가 높을수록 해당 영역의 **어려움이 큰 것**으로 해석합니다.")

    total = len(QUESTIONS)
    step = int(st.session_state.step)
    step = max(0, min(total - 1, step))
    st.session_state.step = step

    # 진행률
    st.caption(f"진행: {step+1}/{total}")
    st.progress((step + 1) / total)

    q = QUESTIONS[step]
    qid = q["id"]

    # 라디오 옵션
    options = [int(s["value"]) for s in SCALE]
    labels = {int(s["value"]): f'{s["value"]} · {s["label"]}' for s in SCALE}

    default_val = int(st.session_state.responses.get(qid, 1))
    default_idx = options.index(default_val) if default_val in options else 0

    # 문항 표시 (questions는 유일화되어 key 충돌 없음)
    val = st.radio(
        f'{qid}. {q["text"]}',
        options=options,
        index=default_idx,
        format_func=lambda v: labels[v],
        horizontal=True,
        key=f"q_{qid}",
    )
    st.session_state.responses[qid] = int(val)

    # 네비게이션 버튼
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

    with b1:
        if st.button("처음으로"):
            st.session_state.step = 0
            st.rerun()

    with b2:
        if st.button("이전", disabled=(step == 0)):
            st.session_state.step = max(0, step - 1)
            st.rerun()

    with b3:
        if st.button("다음", disabled=(step >= total - 1)):
            st.session_state.step = min(total - 1, step + 1)
            st.rerun()

    with b4:
        # 마지막 문항에서만 제출 가능
        can_submit = (step == total - 1) and ok_profile
        if st.button("제출", type="primary", disabled=(not can_submit)):
            # 누락 응답 체크
            ok_answers, missing = validate_all_answered(st.session_state.responses)
            if not ok_answers:
                st.error(f"아직 응답하지 않은 문항이 있습니다. (예: {missing[:5]})")
                st.stop()

            # 요인 점수 계산 (factors.json 그대로 사용)
            factor_scores_dc = []
            factor_scores = score_factors(st.session_state.responses)

            # score_factors가 dataclass(또는 유사)를 반환할 수 있으므로 안전 변환
            for fs in factor_scores:
                if hasattr(fs, "__dict__") and "name" in fs.__dict__:
                    d = {"factorName": fs.name, "mean": fs.mean, "percent": fs.percent, "level": fs.level}
                elif isinstance(fs, dict):
                    # 혹시 dict 형태라면
                    d = {"factorName": fs.get("name") or fs.get("factorName"), "mean": fs.get("mean"), "percent": fs.get("percent"), "level": fs.get("level")}
                else:
                    # 마지막 안전장치
                    d = {"factorName": str(fs), "mean": 0, "percent": 0, "level": "낮음"}
                factor_scores_dc.append(d)

            # 상위 요인(어려움 큰 순)
            top_factors_dc = factor_scores_dc[:2]
            top_factor_names = [t["factorName"] for t in top_factors_dc if t.get("factorName")]

            # 추천(고도화: 코드+태그+다양성)
            recommended_ids = recommend_scenarios_advanced(top_factor_names, k=3)

            # DB 저장
            rid = insert_assessment(
                user_id=USER_ID,
                user_name=st.session_state.profile["name"],
                user_age=int(st.session_state.profile["age"]),
                user_gender=st.session_state.profile["gender"],
                responses=st.session_state.responses,
                factor_scores=factor_scores_dc,
                top_factors=top_factors_dc,
                recommended_ids=recommended_ids,
            )

            st.success(f"저장 완료: 검사 #{rid}")
            st.info("상단의 '결과/추천' 탭에서 확인하세요.")


# -----------------------------
# 탭 2: 결과/추천
# -----------------------------
with tab2:
    st.subheader("최근 결과")
    latest = latest_assessment(USER_ID)
    if not latest:
        st.warning("저장된 결과가 없습니다. 먼저 '검사(1문항씩)' 탭에서 제출해주세요.")
    else:
        st.write(f"저장 시간: {latest['created_at']}")
        if latest.get("user_name"):
            st.caption(f"사용자: {latest.get('user_name')} / {latest.get('user_age')}세 / {latest.get('user_gender')}")

        st.divider()
        st.markdown("### 6요인 점수 (점수가 높을수록 어려움이 큼)")
        for f in latest["factor_scores"]:
            st.write(f"**{f['factorName']}** — 평균 {f['mean']}점 · {f['percent']}/100 · {f['level']}")
            st.progress(min(max(int(f["percent"]), 0), 100))
            st.caption(level_comment(f["level"]))
            st.write("")

        st.divider()
        st.markdown("### 추천 시나리오 (코드+태그 기반, 다양성 고려)")
        rec_ids = latest["recommended_ids"] or []
        if not rec_ids:
            st.info("추천 시나리오가 없습니다.")
        else:
            for sid in rec_ids:
                s = SCENARIOS.get(sid)
                if not s:
                    continue
                title = s.get("title", sid)
                code = s.get("code", "")
                with st.expander(f"{title}  (코드: {code})", expanded=False):
                    tags = s.get("tags") or []
                    if tags:
                        st.caption(" ".join([f"#{t}" for t in tags[:12]]))

                    st.markdown("#### 대사")
                    for d in s.get("dialogues", []):
                        speaker = d.get("speaker") or "대사"
                        st.write(f"- **{speaker}**: {d.get('text','')}")

                    st.divider()
                    st.markdown("#### 평가문항")
                    st.write(s.get("question", ""))

                    opts = s.get("options", [])
                    labels = [o["label"] for o in opts]
                    label_to_opt = {o["label"]: o for o in opts}

                    choice = st.radio("선택", options=labels, horizontal=True, key=f"choose_{sid}")
                    if st.button("선택 저장", key=f"save_{sid}"):
                        opt = label_to_opt.get(choice)
                        if opt:
                            insert_attempt(USER_ID, sid, choice, bool(opt.get("is_correct")))
                            st.success("선택 결과가 저장되었습니다.")
                            st.write("**정답**" if opt.get("is_correct") else "**오답**")
                            st.write(opt.get("rationale") or "해설이 없습니다.")


# -----------------------------
# 탭 3: 내 기록
# -----------------------------
with tab3:
    st.subheader("내 기록")
    st.caption("로그인 없이 테스트용으로, 현재 세션의 사용자ID 기준으로 저장된 기록을 보여줍니다.")
    rows = list_assessments(USER_ID, limit=30)
    if not rows:
        st.info("저장된 기록이 없습니다.")
    else:
        for r in rows:
            top = r["top_factors"] or []
            top_text = ", ".join([f"{t['factorName']}({t['mean']}점/{t['level']})" for t in top])
            profile_text = ""
            if r.get("user_name"):
                profile_text = f" · {r.get('user_name')} / {r.get('user_age')}세 / {r.get('user_gender')}"
            st.write(f"- 검사 #{r['id']} · {r['created_at']}{profile_text} · 상위 영역: {top_text}")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("응답 초기화(처음부터 다시)"):
            st.session_state.step = 0
            st.session_state.responses = {}
            st.rerun()
    with c2:
        if st.button("프로필 초기화"):
            st.session_state.profile = {"name": "", "age": 10, "gender": "선택안함"}
            st.rerun()
