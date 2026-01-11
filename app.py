# app.py (전체 교체본: 첫 화면(Home) + ID/나이/성별 입력 + 시작 버튼)
# 실행: streamlit run app.py

from __future__ import annotations

import json
import sqlite3
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

    # (요청사항) 나이/성별 저장 컬럼 추가
    existing_cols = [r[1] for r in con.execute("PRAGMA table_info(assessments);").fetchall()]
    if "user_age" not in existing_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN user_age INTEGER;")
    if "user_gender" not in existing_cols:
        con.execute("ALTER TABLE assessments ADD COLUMN user_gender TEXT;")

    con.commit()
    con.close()


def now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def insert_assessment(
    user_id: str,
    user_age: int,
    user_gender: str,
    responses: Dict[str, int],
    factor_scores: List[Dict[str, Any]],
    top_factors: List[Dict[str, Any]],
    recommended_ids: List[str],
) -> int:
    con = connect_db()

    cols = [r[1] for r in con.execute("PRAGMA table_info(assessments);").fetchall()]
    has_profile = all(c in cols for c in ["user_age", "user_gender"])

    created_at = now_iso()

    if has_profile:
        cur = con.execute(
            """
            INSERT INTO assessments
            (user_id, created_at, user_age, user_gender,
             responses_json, factor_scores_json, top_factors_json, recommended_scenarios_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                int(user_age),
                str(user_gender),
                json.dumps(responses, ensure_ascii=False),
                json.dumps(factor_scores, ensure_ascii=False),
                json.dumps(top_factors, ensure_ascii=False),
                json.dumps(recommended_ids, ensure_ascii=False),
            ),
        )
    else:
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
        SELECT id, created_at, user_age, user_gender,
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
        "user_age": row[2],
        "user_gender": row[3],
        "factor_scores": json.loads(row[4]),
        "top_factors": json.loads(row[5]),
        "recommended_ids": json.loads(row[6]),
    }


def list_assessments(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    con = connect_db()
    rows = con.execute(
        """
        SELECT id, created_at, user_age, user_gender, top_factors_json
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
                "user_age": r[2],
                "user_gender": r[3],
                "top_factors": json.loads(r[4]),
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
FACTOR_TO_TAG_KEYWORDS: Dict[str, List[str]] = {
    "사회인지·맥락이해": ["눈치", "의도", "뉘앙스", "농담", "표정", "목소리"],
    "상호작용 기술·비언어": ["눈맞춤", "표정", "말투", "거리", "자세", "반응"],
    "사회적 동기·회피/불안": ["모임", "친구", "학교", "대화", "참여", "어울림", "부탁"],
    "반복·집착/감각·경직": ["변화", "계획", "일정", "예민", "습관", "집착"],
    "자기표현·의사소통 유연성": ["부탁", "설명", "의견", "질문", "대답", "대화"],
    "정서·자기조절": ["불안", "긴장", "감정", "조절", "짜증", "진정"],
}


def score_scenario_for_user(scenario: dict, top_factor_names: List[str]) -> int:
    s_code = str(scenario.get("code", "")).strip()
    s_tags = scenario.get("tags") or []
    s_title = str(scenario.get("title", "")).strip()
    s_detail = str(scenario.get("detail", "")).strip()
    blob = " ".join([s_title, s_detail, " ".join(map(str, s_tags))])

    target_codes = set()
    for fn in top_factor_names[:2]:
        for c in FACTOR_TO_CODES.get(fn, []):
            target_codes.add(c)

    score = 0
    if s_code and s_code in target_codes:
        score += 50

    hits = 0
    for fn in top_factor_names[:2]:
        for kw in FACTOR_TO_TAG_KEYWORDS.get(fn, []):
            if kw and kw in blob:
                hits += 1
    score += min(hits, 3) * 10  # 최대 30점
    return score


def pick_with_diversity(scored: List[Tuple[int, dict]], k: int = 3) -> List[str]:
    picked_ids: List[str] = []
    used_primary_tags = set()

    for s_score, s in scored:
        primary = (s.get("tags") or [""])[0]
        if primary and primary in used_primary_tags:
            continue
        picked_ids.append(s["id"])
        if primary:
            used_primary_tags.add(primary)
        if len(picked_ids) >= k:
            return picked_ids

    for s_score, s in scored:
        if s["id"] in picked_ids:
            continue
        picked_ids.append(s["id"])
        if len(picked_ids) >= k:
            return picked_ids

    return picked_ids


def recommend_scenarios_advanced(top_factor_names: List[str], k: int = 3) -> List[str]:
    scored: List[Tuple[int, dict]] = [(score_scenario_for_user(s, top_factor_names), s) for s in SCENARIO_LIST]
    scored.sort(key=lambda x: x[0], reverse=True)
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


def validate_profile(user_id_text: str, age: int, gender: str) -> Tuple[bool, str]:
    if not user_id_text.strip():
        return False, "ID를 입력해주세요."
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
st.set_page_config(page_title="버디 플랜", layout="centered")
init_db()

# 앱 상태(첫 화면 -> 검사/결과/기록)
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | app

# 프로필(첫 화면에서 입력)
if "profile" not in st.session_state:
    st.session_state.profile = {"id": "", "age": 10, "gender": "선택안함"}

# 검사 진행 상태
if "step" not in st.session_state:
    st.session_state.step = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}


# -----------------------------
# CSS: 첫 화면을 첨부 이미지 느낌으로(간단 구현)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 배경 톤 */
.stApp { background: #f6f8fb; }

/* 중앙 카드 */
.bp-wrap { max-width: 920px; margin: 0 auto; padding: 40px 12px; }
.bp-card {
  background: #ffffff;
  border-radius: 22px;
  padding: 34px 26px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.06);
}

/* 타이틀 */
.bp-title { font-size: 52px; font-weight: 900; color: #11a2ff; text-align: center; margin: 0; }
.bp-sub { text-align: center; color: #6b7a90; font-size: 18px; line-height: 1.6; margin-top: 10px; }

/* pill grid */
.bp-pillbox {
  background: #f4f7fb;
  border-radius: 18px;
  padding: 20px;
  margin: 22px 0 18px 0;
}
.bp-pillrow { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
.bp-pill {
  background: #ffffff;
  border: 1px solid #e6eef8;
  color: #2a3a52;
  padding: 12px 18px;
  border-radius: 999px;
  font-weight: 700;
  min-width: 150px;
  text-align: center;
}

/* 입력 박스 */
.bp-form { margin-top: 12px; }
.bp-help { color: #7b8aa3; font-size: 13px; margin-top: 6px; text-align:center; }

/* 시작 버튼 느낌(스트림릿 기본 버튼을 감싸기 때문에 완전 동일하진 않지만 톤은 맞춤) */
div.stButton > button {
  width: 240px;
  height: 58px;
  border-radius: 999px;
  font-size: 18px;
  font-weight: 800;
  background: #11a2ff;
  border: 0;
  color: white;
  box-shadow: 0 12px 22px rgba(17,162,255,0.25);
}
div.stButton > button:hover { background: #0b94ee; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# 1) 첫 화면(Home)
# -----------------------------
if st.session_state.view == "home":
    st.markdown('<div class="bp-wrap"><div class="bp-card">', unsafe_allow_html=True)

    st.markdown('<h1 class="bp-title">버디 플랜</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bp-sub">6가지 주요 사회성 요인을 분석하고,<br/>아이에게 맞는 사회 훈련을 추천해 드립니다.</div>',
        unsafe_allow_html=True,
    )

    # pill 영역(디자인 요소)
    st.markdown('<div class="bp-pillbox"><div class="bp-pillrow">', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">정서 조절</div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">공감 능력</div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">사회적 자기주장</div>', unsafe_allow_html=True)
    st.markdown('</div><div class="bp-pillrow" style="margin-top:10px;">', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">협동성</div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">책임감</div>', unsafe_allow_html=True)
    st.markdown('<div class="bp-pill">사교성</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # 입력 폼
    st.markdown('<div class="bp-form">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        user_id_text = st.text_input("ID", value=st.session_state.profile["id"], placeholder="예: A001 또는 보호자 지정 ID")
    with c2:
        age = st.number_input("나이", min_value=1, max_value=120, value=int(st.session_state.profile["age"]))
    with c3:
        gender = st.selectbox("성별", options=["선택안함", "남", "여", "기타"],
                              index=["선택안함", "남", "여", "기타"].index(st.session_state.profile["gender"]))

    st.session_state.profile = {"id": user_id_text, "age": int(age), "gender": gender}
    ok, msg = validate_profile(user_id_text, int(age), gender)
    if not ok:
        st.markdown(f'<div class="bp-help">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bp-help">입력이 완료되면 아래 버튼을 눌러 평가를 시작하세요.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 시작 버튼
    st.markdown("<div style='display:flex; justify-content:center; margin-top:18px;'>", unsafe_allow_html=True)
    if st.button("평가 시작하기", disabled=not ok):
        # 새 평가 시작: 응답/진행 초기화
        st.session_state.responses = {}
        st.session_state.step = 0
        st.session_state.view = "app"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# -----------------------------
# 2) 본 앱 화면(검사/결과/기록)
# -----------------------------
# 입력 ID를 “사용자 키”로 사용(저장 기준)
USER_ID = st.session_state.profile["id"].strip()

st.title("사회성 6요인 평가")
st.caption("점수가 높을수록 해당 영역의 어려움이 큰 것으로 해석합니다.")

tab1, tab2, tab3 = st.tabs(["검사(1문항씩)", "결과/추천", "내 기록"])


# -----------------------------
# 탭 1: 검사(1문항씩)
# -----------------------------
with tab1:
    st.subheader("검사 진행")
    st.caption(f"ID: {USER_ID} / 나이: {st.session_state.profile['age']} / 성별: {st.session_state.profile['gender']}")

    total = len(QUESTIONS)
    step = int(st.session_state.step)
    step = max(0, min(total - 1, step))
    st.session_state.step = step

    st.caption(f"진행: {step+1}/{total}")
    st.progress((step + 1) / total)

    q = QUESTIONS[step]
    qid = q["id"]

    options = [int(s["value"]) for s in SCALE]
    labels = {int(s["value"]): f'{s["value"]} · {s["label"]}' for s in SCALE}

    default_val = int(st.session_state.responses.get(qid, 1))
    default_idx = options.index(default_val) if default_val in options else 0

    val = st.radio(
        f'{qid}. {q["text"]}',
        options=options,
        index=default_idx,
        format_func=lambda v: labels[v],
        horizontal=True,
        key=f"q_{qid}",  # questions가 유일하므로 충돌 없음
    )
    st.session_state.responses[qid] = int(val)

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
        can_submit = (step == total - 1)
        if st.button("제출", type="primary", disabled=(not can_submit)):
            ok_answers, missing = validate_all_answered(st.session_state.responses)
            if not ok_answers:
                st.error(f"아직 응답하지 않은 문항이 있습니다. (예: {missing[:5]})")
                st.stop()

            factor_scores = score_factors(st.session_state.responses)
            factor_scores_dc: List[Dict[str, Any]] = []
            for fs in factor_scores:
                # logic.py의 FactorScore(dataclass)를 가정
                if hasattr(fs, "__dict__") and "name" in fs.__dict__:
                    factor_scores_dc.append(
                        {"factorName": fs.name, "mean": fs.mean, "percent": fs.percent, "level": fs.level}
                    )
                elif isinstance(fs, dict):
                    factor_scores_dc.append(
                        {
                            "factorName": fs.get("name") or fs.get("factorName"),
                            "mean": fs.get("mean"),
                            "percent": fs.get("percent"),
                            "level": fs.get("level"),
                        }
                    )
                else:
                    factor_scores_dc.append({"factorName": str(fs), "mean": 0, "percent": 0, "level": "낮음"})

            top_factors_dc = factor_scores_dc[:2]
            top_factor_names = [t["factorName"] for t in top_factors_dc if t.get("factorName")]

            recommended_ids = recommend_scenarios_advanced(top_factor_names, k=3)

            rid = insert_assessment(
                user_id=USER_ID,
                user_age=int(st.session_state.profile["age"]),
                user_gender=str(st.session_state.profile["gender"]),
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
        st.warning("저장된 결과가 없습니다. 먼저 검사 탭에서 제출해주세요.")
    else:
        st.write(f"저장 시간: {latest['created_at']}")
        st.caption(f"ID: {USER_ID} / 나이: {latest.get('user_age')} / 성별: {latest.get('user_gender')}")

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
    rows = list_assessments(USER_ID, limit=30)
    if not rows:
        st.info("저장된 기록이 없습니다.")
    else:
        for r in rows:
            top = r["top_factors"] or []
            top_text = ", ".join([f"{t['factorName']}({t['mean']}점/{t['level']})" for t in top])
            st.write(
                f"- 검사 #{r['id']} · {r['created_at']} · 나이:{r.get('user_age')} · 성별:{r.get('user_gender')} · 상위 영역: {top_text}"
            )

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("처음 화면으로"):
            st.session_state.view = "home"
            st.rerun()
    with c2:
        if st.button("응답 초기화(다시 검사)"):
            st.session_state.step = 0
            st.session_state.responses = {}
            st.rerun()
