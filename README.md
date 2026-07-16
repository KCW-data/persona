# 사회조사형 페르소나 시뮬레이션 앱

2026.7.24 강의 실습용 Streamlit 앱입니다. 인구통계학적 정보, 정치 관심, 제도 신뢰, 사회 현안 태도, 정보 이용 방식을 입력하고 Gemini API로 페르소나를 분석합니다.

## 실행

```bash
python3 -m streamlit run lecture_materials_v2/streamlit_app/persona_lab_app.py
```

## 핵심 흐름

1. 이메일을 입력해 실습 세션을 만든다.
2. 인구통계학적 정보와 사회조사형 문항에 답한다.
3. 서버에 설정된 Gemini API로 페르소나를 생성한다. 키가 없거나 호출에 실패하면 결과를 만들지 않고 관리자 설정을 요청한다.
4. AI 규제, 청년 고용, 주거비, 기후위기, 개인정보 등 10개 현안을 확인한다.
5. 페르소나가 각 현안에 대해 1~5점 척도로 응답한다.
6. 사용자가 각 응답을 맞다/틀리다로 검증하고, 틀린 경우 실제 내 응답 점수와 정정 이유를 저장한다.
7. 개인 보고서와 전체 CSV 기록을 다운로드한다.

## 문항 설계 축

- 인구통계학적 정보: 연령대, 성별, 생활권, 교육 상태, 거주 형태, 주관적 경제적 위치
- 정치·사회 관심: 정치 성향 자기배치, 정치 관심, 뉴스 확인 빈도, 공론장 참여
- 제도 신뢰: 정부, 지방정부, 사법제도, 언론, 전문가, 대학·연구기관
- 사회 현안 태도: 기후위기, AI 규제, 청년 고용, 주거비, 불평등, 성평등, 다문화, 개인정보, 지역소멸, 플랫폼 노동
- 정보 이용: 신뢰하는 정보 출처와 출처 검증 성향

## Gemini API와 관리자 대시보드

일반 사용자는 Gemini API 키를 입력하지 않습니다. 강의자는 Streamlit Cloud Secrets 또는 관리자 대시보드에서 키를 설정합니다.

중요: `http://localhost:8501`로 실행 중인 로컬 앱은 Streamlit Cloud Secrets를 읽지 않습니다. Cloud Secrets는 배포된 Streamlit Cloud 주소에서만 적용됩니다. 로컬 테스트에서는 아래 중 하나를 사용합니다.

- 관리자 대시보드에서 런타임 키 등록
- 환경변수 `GEMINI_API_KEY`, `ADMIN_PASSWORD` 설정
- `lecture_materials_v2/streamlit_app/.streamlit/secrets.toml` 파일 생성

Streamlit Cloud에 배포할 때는 Secrets에 아래처럼 등록합니다.

```toml
ADMIN_PASSWORD = "CHANGE_ME_ADMIN_PASSWORD"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3.5-flash"
```

API 키는 GitHub에 커밋하지 않습니다.

관리자 대시보드는 5번 탭에서 비밀번호 로그인 후 확인할 수 있습니다. 관리자 대시보드에서는 Gemini 키 등록/교체, 모델 선택, 전체 응답 기록 확인, CSV 다운로드를 수행합니다.

## 저장 파일

- `data/persona_records.jsonl`: 입력값, 페르소나 분석 결과, 분석 축 기록
- `data/evaluations.csv`: 현안 질문, 페르소나 응답, 사용자 판정, 판정 이유 기록

`data/` 폴더는 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다.

## 배포

GitHub와 Streamlit Cloud 연동 절차는 `GITHUB_STREAMLIT_DEPLOY.md`를 확인하세요.

## 주의

이 앱은 수업 실습용입니다. 결과는 사회조사형 자기응답을 바탕으로 한 가설이며 실제 조사 결과, 성격 진단, 투표 선택 예측, 정치적 설득 도구가 아닙니다.
