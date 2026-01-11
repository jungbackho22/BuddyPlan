# 사회성 6요인 검사 (Python/Streamlit + SQLite 저장)

- 30문항(1~6점) 응답
- 6요인 점수 산출(점수가 높을수록 어려움이 큼)
- 결과 기반 사회적 시나리오 추천(대사 → 평가문항 → 해설)
- 저장 있음(SQLite): 검사 결과 + 시나리오 선택 기록

---

## 1) Mac 기준 실행 방법(초보자용)

### 1-1. Python 설치 확인
터미널에서:
```bash
python3 --version
```
- 3.10 이상이면 OK

### 1-2. 프로젝트 폴더로 이동
```bash
cd sociality-python-app
```

### 1-3. 가상환경 만들기/켜기(권장)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1-4. 패키지 설치
```bash
pip install -r requirements.txt
```

### 1-5. 앱 실행
```bash
streamlit run app.py
```

브라우저가 자동으로 열리고, 보통 아래 주소로 접속됩니다:
- http://localhost:8501

---

## 2) 저장(DB)
- `db/app.db` 파일에 저장됩니다.
- 사용자는 “이 브라우저에서 생성된 임의 사용자ID”로 구분합니다(로그인 없이 간단히).

---

## 3) 커스터마이징(초보자용)
- 추천 규칙(요인→시나리오 코드 매핑)을 바꾸려면 `logic.py`의 `FACTOR_TO_CODES`를 수정하세요.
- 낮음/중간/높음 기준을 바꾸려면 `logic.py`의 `level_from_mean()`을 수정하세요.
