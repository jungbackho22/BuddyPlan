from __future__ import annotations
import json
import uuid
from pathlib import Path
import streamlit as st

from logic import score_factors, recommend_scenarios
from db import init_db, insert_assessment, latest_assessment, list_assessments, insert_attempt

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Buddy-Plan", layout="centered")

def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

QUESTIONS = load_json("questions.json")["questions"]
SCALE = load_json("questions.json")["scale"]
SCENARIOS = {s["id"]: s for s in load_json("scenarios.json")["scenarios"]}

init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
USER_ID = st.session_state.user_id

def level_comment(level: str) -> str:
    if level == "낮음":
        return "현재 이 영역에서의 어려움이 비교적 적게 보고되었습니다."
    if level == "중간":
        return "상황에 따라 어려움이 나타날 수 있어, 시나리오 연습을 권장합니다."
    return "이 영역에서의 어려움이 비교적 크게 보고되었습니다. 추천 시나리오로 단계적으로 연습해보세요."

st.title("Buddy-Plan")
st.caption("30문항을 통해 6가지 주요 사회성 요인을 분석하고,당신에게 맞는 사회적 시나리오를 추천해 드립니다.")

tab1, tab2, tab3 = st.tabs(["검사", "결과/추천", "내 기록"])

with tab1:
    st.subheader("검사 응답")
    st.write("점수가 높을수록 해당 영역의 **어려움이 큰 것**으로 해석합니다.")
    st.divider()

    if "responses" not in st.session_state:
        st.session_state.responses = {}

    for q in QUESTIONS:
        qid = q["id"]
        default = st.session_state.responses.get(qid, 1)

        options = [s["value"] for s in SCALE]
        labels = {s["value"]: f'{s["value"]} · {s["label"]}' for s in SCALE}
        idx = options.index(default) if default in options else 0

        val = st.radio(
            f'{qid}. {q["text"]}',
            options=options,
            index=idx,
            format_func=lambda v: labels[v],
            horizontal=True,
            key=f"q_{qid}",
        )
        st.session_state.responses[qid] = int(val)

    if st.button("제출하고 저장하기", type="primary"):
        if len(st.session_state.responses) != len(QUESTIONS):
            st.error("모든 문항에 응답해주세요.")
        else:
            factor_scores = score_factors(st.session_state.responses)
            top = factor_scores[:2]
            rec_ids = recommend_scenarios([t.name for t in top], max_n=3)

            factor_scores_dict = [
                {"factorName": fs.name, "mean": fs.mean, "percent": fs.percent, "level": fs.level}
                for fs in factor_scores
            ]
            top_dict = factor_scores_dict[:2]

            rid = insert_assessment(USER_ID, st.session_state.responses, factor_scores_dict, top_dict, rec_ids)
            st.success(f"저장 완료: 검사 #{rid}")
            st.info("상단의 '결과/추천' 탭에서 확인하세요.")

with tab2:
    st.subheader("최근 결과")
    latest = latest_assessment(USER_ID)
    if not latest:
        st.warning("저장된 결과가 없습니다. 먼저 '검사' 탭에서 제출해주세요.")
    else:
        st.write(f"저장 시간: {latest['created_at']}")
        st.divider()

        st.markdown("### 6요인 점수(점수가 높을수록 어려움이 큼)")
        for f in latest["factor_scores"]:
            st.write(f"**{f['factorName']}** — 평균 {f['mean']}점 · {f['percent']}/100 · {f['level']}")
            st.progress(min(max(int(f["percent"]), 0), 100))
            st.caption(level_comment(f["level"]))
            st.write("")

        st.divider()
        st.markdown("### 추천 시나리오(3개)")
        rec_ids = latest["recommended_ids"]
        if not rec_ids:
            st.info("추천 시나리오가 없습니다.")
        else:
            for sid in rec_ids:
                s = SCENARIOS.get(sid)
                if not s:
                    continue
                with st.expander(f"{s['title']}  (코드: {s.get('code','')})", expanded=False):
                    tags = s.get("tags") or []
                    if tags:
                        st.caption(" ".join([f"#{t}" for t in tags[:10]]))

                    st.markdown("#### 대사")
                    for d in s.get("dialogues", []):
                        speaker = d.get("speaker") or "대사"
                        st.write(f"- **{speaker}**: {d.get('text','')}")

                    st.divider()
                    st.markdown("#### 평가문항")
                    st.write(s.get("question",""))

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

with tab3:
    st.subheader("내 기록")
    rows = list_assessments(USER_ID)
    if not rows:
        st.info("저장된 기록이 없습니다.")
    else:
        for r in rows:
            top = r["top_factors"]
            top_text = ", ".join([f"{t['factorName']}({t['mean']}점/{t['level']})" for t in top])
            st.write(f"- 검사 #{r['id']} · {r['created_at']} · 상위 영역: {top_text}")
