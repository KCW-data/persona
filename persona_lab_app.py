from __future__ import annotations

import csv
import hmac
import json
import os
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RECORDS_PATH = DATA_DIR / "persona_records.jsonl"
EVALUATIONS_PATH = DATA_DIR / "evaluations.csv"
LOCAL_SECRETS_PATH = APP_DIR / ".streamlit" / "secrets.toml"

DEFAULT_MODEL = "gemini-3.5-flash"
MODEL_OPTIONS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

SURVEY_BASIS = [
    "통계청 사회조사: 인구·가구 특성, 사회참여, 신뢰, 생활여건, 사회적 관심 영역을 참고",
    "한국종합사회조사/KGSS: 정치 관심, 이념 자기배치, 제도 신뢰, 사회적 태도 문항 구조를 참고",
    "World Values Survey: 민주주의, 권위, 다양성, 환경, 자기표현 가치 축을 참고",
    "European Social Survey: 정치, 미디어 이용, 신뢰, 기후변화, 사회인구학 문항 구조를 참고",
]

REGIONS = [
    "서울·수도권",
    "광주·전라권",
    "대전·충청권",
    "대구·경북권",
    "부산·울산·경남권",
    "강원·제주권",
    "응답하지 않음",
]

GENDER_OPTIONS = ["여성", "남성", "논바이너리/기타", "응답하지 않음"]
EDUCATION_OPTIONS = ["대학 재학", "대학 졸업", "대학원 재학 이상", "기타", "응답하지 않음"]
ECONOMIC_OPTIONS = ["매우 낮음", "낮은 편", "중간", "높은 편", "매우 높음", "응답하지 않음"]
HOUSING_OPTIONS = ["기숙사·자취", "가족과 거주", "자가", "전월세", "기타", "응답하지 않음"]

INFO_SOURCES = [
    "논문·학술자료",
    "정부·공공기관 통계",
    "언론 기사",
    "전문가 인터뷰",
    "유튜브·팟캐스트",
    "커뮤니티·SNS",
    "주변 사람의 경험",
]

TRUST_ITEMS = [
    ("central_government_trust", "중앙정부를 신뢰한다"),
    ("local_government_trust", "지방정부를 신뢰한다"),
    ("court_trust", "법원·사법제도를 신뢰한다"),
    ("press_trust", "언론을 신뢰한다"),
    ("science_trust", "과학자·전문가 집단을 신뢰한다"),
    ("university_trust", "대학과 연구기관을 신뢰한다"),
]

ISSUE_ITEMS = [
    ("climate_transition", "기후위기 대응을 위해 에너지 전환과 생활 변화가 필요하다"),
    ("ai_regulation", "AI 활용이 늘어날수록 개인정보, 저작권, 차별 문제를 강하게 규제해야 한다"),
    ("youth_jobs", "청년 고용 문제는 개인 노력보다 산업 구조와 제도 설계의 영향이 크다"),
    ("housing_policy", "주거비 문제는 시장 자율보다 공공 정책 개입이 더 필요하다"),
    ("inequality", "소득·자산 격차 완화는 사회 안정에 중요한 과제다"),
    ("gender_equality", "성평등과 다양성 정책은 계속 확대되어야 한다"),
    ("multicultural", "이주민·다문화 구성원의 사회 참여를 더 적극적으로 보장해야 한다"),
    ("privacy", "공공안전이나 편의보다 개인정보 보호를 더 우선해야 한다"),
    ("regional_decline", "지역소멸과 수도권 집중은 개인 선택보다 국가적 대응이 필요한 문제다"),
    ("platform_labor", "플랫폼 노동과 프리랜서 노동에 대한 사회보장 장치가 필요하다"),
]

LIKERT_LABELS = {
    1: "1점 매우 반대",
    2: "2점 반대",
    3: "3점 중립/판단 유보",
    4: "4점 찬성",
    5: "5점 매우 찬성",
}

CURRENT_ISSUES = {
    "AI 규제와 저작권": "생성형 AI가 만든 결과물의 저작권과 학습 데이터 사용을 어떻게 규제해야 할까?",
    "청년 고용과 포트폴리오": "채용에서 AI 활용 경험을 어떤 기준으로 평가하는 것이 공정할까?",
    "주거비와 청년 삶": "청년 주거비 부담을 줄이기 위해 공공 정책은 어디까지 개입해야 할까?",
    "기후위기와 에너지 비용": "탄소 감축을 위해 에너지 비용 상승을 어느 정도까지 받아들일 수 있을까?",
    "개인정보와 공공안전": "범죄 예방이나 행정 효율을 위해 개인정보 활용을 확대해도 될까?",
    "지역소멸과 대학": "지역 대학과 청년 인재가 지역소멸 문제 해결에 어떤 역할을 해야 할까?",
    "다문화와 사회통합": "이주민과 다문화 구성원의 사회 참여를 확대하기 위해 무엇이 필요할까?",
    "플랫폼 노동": "배달, 프리랜서, 플랫폼 노동자의 권리 보장을 어떻게 설계해야 할까?",
    "저출생과 돌봄": "저출생 문제를 개인 선택이 아니라 사회 구조 문제로 다루려면 무엇이 바뀌어야 할까?",
    "대학 등록금과 교육 기회": "대학 등록금과 교육비 부담을 줄이기 위해 공공 지원을 확대해야 할까?",
}

ISSUE_ANCHOR_MAP = {
    "AI 규제와 저작권": ["ai_regulation", "privacy"],
    "청년 고용과 포트폴리오": ["youth_jobs", "platform_labor"],
    "주거비와 청년 삶": ["housing_policy", "inequality"],
    "기후위기와 에너지 비용": ["climate_transition"],
    "개인정보와 공공안전": ["privacy", "ai_regulation"],
    "지역소멸과 대학": ["regional_decline", "youth_jobs"],
    "다문화와 사회통합": ["multicultural"],
    "플랫폼 노동": ["platform_labor", "inequality"],
    "저출생과 돌봄": ["youth_jobs", "inequality", "housing_policy"],
    "대학 등록금과 교육 기회": ["inequality", "youth_jobs"],
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def normalize_email(email: str) -> str:
    return email.strip().lower()


def safe_filename_part(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text).strip("_") or "user"


def secret_value_and_source(name: str) -> tuple[str, str]:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    if value:
        return str(value), "Streamlit Cloud Secrets"

    if LOCAL_SECRETS_PATH.exists():
        try:
            local_secrets = tomllib.loads(LOCAL_SECRETS_PATH.read_text(encoding="utf-8"))
            value = local_secrets.get(name, "")
        except (tomllib.TOMLDecodeError, OSError):
            value = ""
        if value:
            return str(value), "로컬 secrets.toml"

    value = os.environ.get(name, "") or ""
    if value:
        return str(value), "환경변수"
    return "", "미설정"


def safe_secret(name: str) -> str:
    value, _ = secret_value_and_source(name)
    return value


def configured_admin_password() -> str:
    return safe_secret("ADMIN_PASSWORD")


def admin_login(password: str) -> bool:
    expected = configured_admin_password()
    return bool(expected) and hmac.compare_digest(password, expected)


def gemini_runtime_config() -> dict[str, str]:
    key = safe_secret("GEMINI_API_KEY")
    model = safe_secret("GEMINI_MODEL") or DEFAULT_MODEL
    return {"GEMINI_API_KEY": key, "GEMINI_MODEL": model}


def gemini_status_label() -> str:
    if gemini_runtime_config()["GEMINI_API_KEY"]:
        return "분석 준비 완료"
    return "분석 준비 필요"


def gemini_admin_status_label() -> str:
    key, source = secret_value_and_source("GEMINI_API_KEY")
    if key:
        return f"{source} 사용"
    return "Gemini 키 미설정"


def slider_score(value: int) -> int:
    return int((value - 1) / 4 * 100)


def avg(values: list[int]) -> int:
    return round(sum(values) / len(values)) if values else 0


def source_score(sources: list[str]) -> int:
    score = 35
    if "논문·학술자료" in sources:
        score += 25
    if "정부·공공기관 통계" in sources:
        score += 20
    if "전문가 인터뷰" in sources:
        score += 10
    if "언론 기사" in sources:
        score += 5
    if "커뮤니티·SNS" in sources:
        score -= 7
    return max(0, min(100, score))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    new_fields = existing_fields + [key for key in row.keys() if key not in existing_fields]
    if new_fields != existing_fields:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
    else:
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow(row)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_evaluations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if path.exists():
            path.unlink()
        return
    df.to_csv(path, index=False, encoding="utf-8")


def profile_email(record: dict[str, Any]) -> str:
    profile = record.get("profile", {})
    return str(profile.get("email") or profile.get("email_hash") or "")


def record_email_values(records: list[dict[str, Any]], evals: pd.DataFrame) -> list[str]:
    values = {profile_email(record) for record in records if profile_email(record)}
    for col in ["email", "email_hash"]:
        if col in evals.columns:
            values.update(str(value) for value in evals[col].dropna().tolist() if str(value))
    return sorted(values)


def delete_data_for_email(email_value: str) -> tuple[int, int]:
    records = load_jsonl(RECORDS_PATH)
    remaining_records = [record for record in records if profile_email(record) != email_value]
    deleted_records = len(records) - len(remaining_records)
    write_jsonl(RECORDS_PATH, remaining_records)

    evals = load_evaluations(EVALUATIONS_PATH)
    deleted_evals = 0
    if not evals.empty:
        mask = pd.Series(False, index=evals.index)
        for col in ["email", "email_hash"]:
            if col in evals.columns:
                mask = mask | (evals[col].astype(str) == email_value)
        deleted_evals = int(mask.sum())
        write_dataframe_csv(EVALUATIONS_PATH, evals.loc[~mask].copy())
    return deleted_records, deleted_evals


def delete_all_data() -> tuple[int, int]:
    record_count = len(load_jsonl(RECORDS_PATH))
    eval_count = len(load_evaluations(EVALUATIONS_PATH))
    for path in [RECORDS_PATH, EVALUATIONS_PATH]:
        if path.exists():
            path.unlink()
    return record_count, eval_count


def latest_persona(email: str) -> dict[str, Any] | None:
    records = [
        row
        for row in load_jsonl(RECORDS_PATH)
        if row.get("profile", {}).get("email") == email and row.get("analysis_method") == "gemini"
    ]
    if not records:
        return None
    return records[-1]


def compute_axes(profile: dict[str, Any]) -> dict[str, int]:
    issues = profile["issue_attitudes"]
    trust = profile["institutional_trust"]
    ideology = profile["political_ideology"]
    ideology_center_distance = abs(ideology - 5)

    social_reform = avg(
        [
            slider_score(issues["inequality"]),
            slider_score(issues["gender_equality"]),
            slider_score(issues["multicultural"]),
            slider_score(issues["platform_labor"]),
        ]
    )
    policy_intervention = avg(
        [
            slider_score(issues["housing_policy"]),
            slider_score(issues["youth_jobs"]),
            slider_score(issues["regional_decline"]),
            slider_score(issues["climate_transition"]),
        ]
    )
    tech_caution = avg([slider_score(issues["ai_regulation"]), slider_score(issues["privacy"])])
    institutional_trust = avg([slider_score(v) for v in trust.values()])
    evidence_orientation = avg(
        [
            source_score(profile["information_sources"]),
            slider_score(profile["evidence_check"]),
            slider_score(profile["news_frequency"]),
        ]
    )
    civic_participation = avg(
        [
            slider_score(profile["political_interest"]),
            slider_score(profile["community_participation"]),
            slider_score(profile["public_discussion_comfort"]),
        ]
    )

    return {
        "사회개혁 지향": social_reform,
        "정책개입 선호": policy_intervention,
        "기술규제 신중성": tech_caution,
        "제도 신뢰": institutional_trust,
        "근거 검증 성향": evidence_orientation,
        "공론장 참여 성향": civic_participation,
        "이념 선명도": round(ideology_center_distance / 5 * 100),
    }


def rule_based_persona(profile: dict[str, Any]) -> dict[str, Any]:
    axes = compute_axes(profile)
    top_axis = max(axes, key=axes.get)
    ideology = profile["political_ideology"]

    if axes["근거 검증 성향"] >= 72 and axes["기술규제 신중성"] >= 65:
        persona_type = "근거 검증형 기술 신중론자"
        summary = "새 기술의 효용을 인정하되, 개인정보·저작권·차별 같은 부작용을 근거 중심으로 확인하려는 경향이 강하다."
    elif axes["사회개혁 지향"] >= 70 and axes["정책개입 선호"] >= 65:
        persona_type = "사회문제 해결형 정책 지지자"
        summary = "불평등, 주거, 고용, 지역 문제를 개인 책임보다 사회 구조와 제도 설계의 문제로 보는 경향이 있다."
    elif axes["제도 신뢰"] < 45 and axes["공론장 참여 성향"] >= 55:
        persona_type = "비판적 참여형 관찰자"
        summary = "제도에 대한 신뢰는 높지 않지만 사회 현안에 관심을 갖고 토론과 검증을 통해 판단하려는 경향이 있다."
    elif axes["정책개입 선호"] < 45 and ideology >= 6:
        persona_type = "자율 중시형 현실주의자"
        summary = "사회 문제 해결에서 공공 개입의 필요성을 인정하더라도, 비용과 책임 배분을 신중하게 따지는 경향이 있다."
    else:
        persona_type = "균형 탐색형 사회 관찰자"
        summary = "특정 입장을 강하게 단정하기보다 이슈별 근거와 조건을 비교하며 판단하려는 경향이 있다."

    evidence = [
        f"가장 두드러진 축: {top_axis} {axes[top_axis]}점",
        f"정치 성향 자기배치: {ideology}/10",
        f"정치 관심: {profile['political_interest']}/5",
        f"주요 정보 출처: {', '.join(profile['information_sources']) or '미입력'}",
        f"사회 현안 자유응답: {profile['open_issue_note'] or '미입력'}",
    ]

    uncertainty = [
        "이 분석은 사회조사형 자기응답을 바탕으로 한 가설이며 실제 성격 진단이 아니다.",
        "정당 선호, 투표 선택, 민감한 개인 신념은 입력하지 않았으므로 추정하지 않는다.",
        "문항 수가 제한되어 있어 현안별 실제 판단은 질문 맥락에 따라 달라질 수 있다.",
    ]

    return {
        "analysis_method": "rule_based_fallback",
        "persona_type": persona_type,
        "summary": summary,
        "axes": axes,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "llm_json": {},
        "profile": profile,
    }


def compact_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, indent=2)


def persona_prompt(profile: dict[str, Any]) -> str:
    return f"""
당신은 사회조사 방법론과 페르소나 기반 사회시뮬레이션을 설명하는 연구 보조자다.
아래 응답은 수업 실습용 자기응답이다. 이 자료를 바탕으로 사용자의 사회 현안 판단 페르소나를 분석하라.

중요 원칙:
1. 실제 성격 진단, 정신상태 진단, 투표 선택 예측, 정당 지지 추정은 하지 않는다.
2. 입력하지 않은 민감정보를 단정하지 않는다.
3. 사회조사형 문항에서 확인되는 경향, 근거, 불확실성을 분리한다.
4. 분석은 한국어로 작성한다.
5. JSON만 출력한다. 설명 문장, 마크다운 코드펜스, 불필요한 주석을 붙이지 않는다.

참고한 문항 설계 축:
{json.dumps(SURVEY_BASIS, ensure_ascii=False, indent=2)}

사용자 응답:
{compact_profile(profile)}

출력 JSON 스키마:
{{
  "persona_title": "짧은 페르소나 이름",
  "one_sentence": "한 문장 요약",
  "detailed_summary": "6~8문장 분석. 인구통계, 정치 관심, 제도 신뢰, 사회 현안 태도, 정보 출처를 함께 해석",
  "axes": [
    {{"name": "축 이름", "score": 0-100, "interpretation": "해석", "evidence": ["입력 근거1", "입력 근거2"]}}
  ],
  "likely_responses": [
    {{"issue": "현안명", "likely_position": "예상 반응", "reason": "근거", "uncertainty": "불확실성"}}
  ],
  "validation_questions": ["추가로 확인해야 할 질문1", "질문2", "질문3"],
  "do_not_infer": ["추정하면 안 되는 정보1", "정보2"],
  "portfolio_sentence": "이 실습을 포트폴리오에 설명하는 한 문장"
}}
""".strip()


def issue_anchor_context(persona: dict[str, Any], issues: dict[str, str]) -> list[dict[str, Any]]:
    profile = persona.get("profile", {})
    attitudes = profile.get("issue_attitudes", {})
    context = []
    for issue_name, question in issues.items():
        fields = ISSUE_ANCHOR_MAP.get(issue_name, [])
        values = []
        for field in fields:
            if field in attitudes:
                values.append({"field": field, "score_1_to_5": attitudes[field]})
        context.append(
            {
                "issue_name": issue_name,
                "question": question,
                "related_input_scores": values,
            }
        )
    return context


def issue_prompt(persona: dict[str, Any], issues: dict[str, str]) -> str:
    return f"""
당신은 아래 페르소나 분석 결과를 바탕으로 실제 사회조사에 참여한 응답자처럼 답한다.
목표는 "이 페르소나가 실제 사람처럼 10개 현안에 어떻게 응답할지"를 시뮬레이션하고, 이후 사용자가 맞다/틀리다로 검증하게 하는 것이다.
따라서 분석 보고서처럼 회피하지 말고, 실제 설문 응답자처럼 1~5점 중 하나를 선택한다.
정당 지지, 투표 선택, 입력되지 않은 민감정보는 새로 추정하지 않는다.
응답은 한국어 JSON만 출력한다.
각 현안에 대해 1~5점 리커트 척도로 답한다.

척도:
1 = 매우 반대
2 = 반대
3 = 중립/판단 유보
4 = 찬성
5 = 매우 찬성

응답 원칙:
- 출력 문장은 실제 설문 응답자의 답변처럼 쓴다. 메타 설명이나 분석 보고서 표현을 반복하지 않는다.
- 3점은 정말 판단이 혼합되거나 입력 근거가 약할 때만 사용한다.
- 10개 응답 중 3점은 최대 2개까지만 허용한다.
- 10개 모두 같은 점수, 특히 모두 3점은 실패다.
- related_input_scores의 1~5점 입력값은 가장 강한 앵커다. 특별한 반대 근거가 없으면 같은 방향으로 응답한다.
- 입력값이 4 또는 5이면 원칙적으로 4~5점, 입력값이 1 또는 2이면 원칙적으로 1~2점으로 응답한다.
- 여러 입력값이 충돌하면 정치 관심, 제도 신뢰, 근거 검증 성향, 상세 페르소나 요약을 함께 고려해 하나를 고른다.
- 각 현안에 대해 실제 응답자처럼 1인칭 응답 문장을 작성한다. 예: "나는 공공 개입이 더 필요하다고 보는 편이다."

페르소나:
{json.dumps(persona, ensure_ascii=False, indent=2)}

현안 목록:
{json.dumps([{"issue_name": key, "question": value} for key, value in issues.items()], ensure_ascii=False, indent=2)}

현안별 입력 문항 앵커:
{json.dumps(issue_anchor_context(persona, issues), ensure_ascii=False, indent=2)}

출력 JSON 스키마:
{{
  "overall_pattern": "10개 현안 응답의 전체 경향 2~3문장",
  "responses": [
    {{
      "issue_name": "현안명",
      "question": "질문",
      "score": 1,
      "label": "1점 매우 반대",
      "response_text": "실제 응답자처럼 쓴 1인칭 응답 문장 1~2문장",
      "reason": "페르소나 입력 근거에 기반한 이유 2~3문장",
      "confidence": "높음/중간/낮음",
      "uncertainty": "현재 입력만으로 부족한 점",
      "verification_hint": "사용자가 맞다/틀리다를 판단할 때 볼 기준"
    }}
  ]
}}

주의:
- responses는 반드시 위 현안 목록과 같은 순서로 정확히 10개를 출력한다.
- score는 반드시 정수 1,2,3,4,5 중 하나다.
- reason에는 입력 근거와 연결되는 설명을 넣는다.
- response_text는 분석문이 아니라 실제 설문 응답자처럼 쓴다.
- JSON 외 텍스트를 출력하지 않는다.
""".strip()


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def normalize_issue_batch(issue_json: dict[str, Any]) -> dict[str, Any]:
    responses = issue_json.get("responses", [])
    if not isinstance(responses, list):
        raise ValueError("responses가 목록이 아닙니다.")
    if len(responses) != len(CURRENT_ISSUES):
        raise ValueError(f"현안 응답 수가 {len(CURRENT_ISSUES)}개가 아닙니다.")

    normalized = []
    expected_items = list(CURRENT_ISSUES.items())
    for idx, (issue_name, question) in enumerate(expected_items):
        item = responses[idx] if idx < len(responses) else {}
        try:
            score = int(item.get("score", 3))
        except (TypeError, ValueError):
            score = 3
        score = max(1, min(5, score))
        normalized.append(
            {
                "issue_name": issue_name,
                "question": question,
                "score": score,
                "label": LIKERT_LABELS[score],
                "response_text": str(item.get("response_text", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": str(item.get("confidence", "")).strip(),
                "uncertainty": str(item.get("uncertainty", "")).strip(),
                "verification_hint": str(item.get("verification_hint", "")).strip(),
            }
        )

    scores = [item["score"] for item in normalized]
    if len(set(scores)) == 1:
        raise ValueError("모든 현안 응답 점수가 동일합니다. 실제 응답자처럼 변별된 응답이 필요합니다.")
    if scores.count(3) > 2:
        raise ValueError("3점 응답이 너무 많습니다. 판단 유보는 최대 2개까지만 허용합니다.")

    return {
        "overall_pattern": str(issue_json.get("overall_pattern", "")).strip(),
        "responses": normalized,
    }


def extract_interaction_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    parts: list[str] = []
    for step in data.get("steps", []):
        for item in step.get("content", []) or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
    if parts:
        return "\n".join(parts)
    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                parts.append(part["text"])
    return "\n".join(parts)


def call_gemini(api_key: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    interactions_payload = {"model": model, "input": prompt}
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers=headers,
            json=interactions_payload,
            timeout=60,
        )
        if response.ok:
            data = response.json()
            text = extract_interaction_text(data)
            if text.strip():
                return text, data
    except requests.RequestException:
        pass

    generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        generate_url,
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return extract_interaction_text(data), data


def build_profile(form: dict[str, Any], email: str) -> dict[str, Any]:
    return {
        "email": email,
        "created_at": now_iso(),
        "demographics": {
            "age_range": form["age_range"],
            "gender": form["gender"],
            "region": form["region"],
            "education": form["education"],
            "housing": form["housing"],
            "subject_area": form["subject_area"],
            "economic_class": form["economic_class"],
        },
        "political_ideology": form["political_ideology"],
        "political_interest": form["political_interest"],
        "news_frequency": form["news_frequency"],
        "community_participation": form["community_participation"],
        "public_discussion_comfort": form["public_discussion_comfort"],
        "evidence_check": form["evidence_check"],
        "information_sources": form["information_sources"],
        "institutional_trust": form["institutional_trust"],
        "issue_attitudes": form["issue_attitudes"],
        "open_issue_note": form["open_issue_note"],
        "future_focus": form["future_focus"],
    }


def normalize_llm_persona(profile: dict[str, Any], llm_json: dict[str, Any] | None, raw_text: str = "") -> dict[str, Any]:
    fallback = rule_based_persona(profile)
    if not llm_json:
        fallback["raw_llm_text"] = raw_text
        return fallback

    axes: dict[str, int] = {}
    for item in llm_json.get("axes", []):
        name = str(item.get("name", "분석 축"))
        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        axes[name] = max(0, min(100, score))
    if not axes:
        axes = fallback["axes"]

    evidence = []
    for item in llm_json.get("axes", []):
        evidence.extend(item.get("evidence", [])[:2])
    if not evidence:
        evidence = fallback["evidence"]

    uncertainty = list(llm_json.get("do_not_infer", [])) + [
        response.get("uncertainty", "")
        for response in llm_json.get("likely_responses", [])
        if response.get("uncertainty")
    ]
    if not uncertainty:
        uncertainty = fallback["uncertainty"]

    return {
        "analysis_method": "gemini",
        "persona_type": llm_json.get("persona_title", fallback["persona_type"]),
        "summary": llm_json.get("one_sentence", fallback["summary"]),
        "detailed_summary": llm_json.get("detailed_summary", ""),
        "axes": axes,
        "evidence": evidence[:8],
        "uncertainty": uncertainty[:8],
        "likely_responses": llm_json.get("likely_responses", []),
        "validation_questions": llm_json.get("validation_questions", []),
        "portfolio_sentence": llm_json.get("portfolio_sentence", ""),
        "llm_json": llm_json,
        "raw_llm_text": raw_text,
        "profile": profile,
    }


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        profile = record.get("profile", {})
        demo = profile.get("demographics", {})
        axes = record.get("axes", {})
        rows.append(
            {
                "created_at": profile.get("created_at", record.get("created_at", "")),
                "email": profile.get("email") or profile.get("email_hash", ""),
                "method": record.get("analysis_method", ""),
                "persona_type": record.get("persona_type", ""),
                "region": demo.get("region", ""),
                "subject_area": demo.get("subject_area", ""),
                "future_focus": profile.get("future_focus", ""),
                **axes,
            }
        )
    return pd.DataFrame(rows)


def persona_report(persona: dict[str, Any], answer: dict[str, Any] | None = None) -> str:
    axes = persona.get("axes", {})
    evidence = "\n".join(f"- {item}" for item in persona.get("evidence", []))
    uncertainty = "\n".join(f"- {item}" for item in persona.get("uncertainty", []))
    lines = [
        "# 사회조사형 페르소나 분석 기록",
        "",
        f"## 분석 방식\n{persona.get('analysis_method', '')}",
        "",
        f"## 페르소나\n{persona.get('persona_type', '')}",
        "",
        f"## 한 문장 요약\n{persona.get('summary', '')}",
        "",
    ]
    if persona.get("detailed_summary"):
        lines.extend(["## 상세 분석", persona["detailed_summary"], ""])
    lines.extend(["## 분석 축", *(f"- {label}: {value}점" for label, value in axes.items()), ""])
    lines.extend(["## 근거 입력", evidence or "- 없음", ""])
    lines.extend(["## 추정하면 안 되는 정보", uncertainty or "- 없음", ""])
    if persona.get("validation_questions"):
        lines.extend(["## 추가 검증 질문", *(f"- {q}" for q in persona["validation_questions"]), ""])
    if answer:
        if answer.get("responses"):
            lines.extend(["## 10개 현안 5점 응답"])
            for item in answer["responses"]:
                lines.extend(
                    [
                        f"### {item.get('issue_name', '')}",
                        f"- 질문: {item.get('question', '')}",
                        f"- 응답: {item.get('label', '')}",
                        f"- 응답 문장: {item.get('response_text', '')}",
                        f"- 이유: {item.get('reason', '')}",
                        f"- 신뢰도: {item.get('confidence', '')}",
                        f"- 불확실성: {item.get('uncertainty', '')}",
                        "",
                    ]
                )
    lines.extend(["## 한계", "이 기록은 수업 실습용 가설이며 실제 조사 결과, 성격 진단, 정치 선택 예측이 아니다."])
    return "\n".join(lines)


def metric_card(label: str, value: int) -> None:
    st.metric(label, f"{value}점")
    st.progress(max(0, min(100, value)) / 100)


def render_persona(persona: dict[str, Any]) -> None:
    st.subheader(persona.get("persona_type", "페르소나"))
    st.write(persona.get("summary", ""))
    if persona.get("analysis_method") == "gemini":
        st.success("LLM 기반 사회조사형 분석 결과입니다.")
    else:
        st.warning("이 기록은 LLM 분석 이전의 예비 기록입니다. 새 분석은 관리자 설정 완료 후 다시 실행하세요.")

    if persona.get("detailed_summary"):
        st.markdown("#### 상세 분석")
        st.write(persona["detailed_summary"])

    axes = persona.get("axes", {})
    cols = st.columns(min(4, max(1, len(axes))))
    for col, (label, value) in zip(cols * 3, axes.items()):
        with col:
            metric_card(label, int(value))

    left, right = st.columns(2)
    with left:
        st.markdown("#### 분석 근거")
        for item in persona.get("evidence", []):
            st.write(f"- {item}")
    with right:
        st.markdown("#### 단정하지 않을 정보")
        for item in persona.get("uncertainty", []):
            st.write(f"- {item}")

    if persona.get("likely_responses"):
        st.markdown("#### 현안별 예상 반응")
        st.dataframe(pd.DataFrame(persona["likely_responses"]), width="stretch")

    if persona.get("validation_questions"):
        st.markdown("#### 추가 검증 질문")
        for question in persona["validation_questions"]:
            st.write(f"- {question}")


st.set_page_config(page_title="사회조사형 페르소나 실습", page_icon="persona", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.4rem; max-width: 1220px; }
    div[data-testid="stMetric"] {
      background: #f7fafb;
      border: 1px solid #d9e3ea;
      border-radius: 8px;
      padding: 14px 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
      border: 1px solid #d9e3ea;
      border-radius: 8px;
      padding: 8px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("사회조사형 페르소나 시뮬레이션")
st.caption("인구통계·정치 관심·제도 신뢰·사회 현안 태도를 입력하고 페르소나 분석 결과와 현안 질문 응답을 검증합니다.")

with st.sidebar:
    st.header("세션")
    email = st.text_input("이메일", value=st.session_state.get("email", ""), placeholder="name@example.com")
    if email and valid_email(email):
        st.session_state["email"] = normalize_email(email)
        st.success(st.session_state["email"])
        if st.button("최근 페르소나 불러오기"):
            loaded = latest_persona(st.session_state["email"])
            if loaded:
                st.session_state["persona"] = loaded
                st.session_state["profile"] = loaded.get("profile", {})
                st.success("최근 페르소나를 불러왔습니다.")
            else:
                st.info("아직 저장된 페르소나가 없습니다.")
    elif email:
        st.warning("이메일 형식을 확인하세요.")

    st.divider()
    st.header("분석 상태")
    runtime = gemini_runtime_config()
    st.write(gemini_status_label())
    st.caption(f"모델: {runtime['GEMINI_MODEL']}")

tab_start, tab_input, tab_persona, tab_issue, tab_admin = st.tabs(
    ["1. 설계", "2. 사회조사 입력", "3. 페르소나 분석", "4. 현안 질문 검증", "5. 관리자 대시보드"]
)

with tab_start:
    st.subheader("문항 설계 기준")
    st.write(
        "이 앱은 단일 기후위기 문항이 아니라 사회조사에서 자주 쓰는 축을 축약해 사용합니다. "
        "인구통계학적 정보, 사회경제적 자기평가, 정치 관심, 이념 자기배치, 제도 신뢰, 미디어 이용, 사회 현안 태도를 함께 입력합니다."
    )
    for source in SURVEY_BASIS:
        st.write(f"- {source}")

    with st.expander("검증 루브릭"):
        st.table(
            pd.DataFrame(
                [
                    {"판정": "맞다", "기준": "페르소나의 5점 응답이 내 실제 입장과 대체로 일치한다."},
                    {"판정": "틀리다", "기준": "페르소나의 5점 응답이 내 실제 입장과 다르다. 실제 내 점수와 정정 이유를 함께 남긴다."},
                ]
            )
        )

with tab_input:
    st.subheader("사회조사형 입력")
    if not valid_email(st.session_state.get("email", "")):
        st.warning("왼쪽 사이드바에서 이메일을 먼저 입력하세요.")

    with st.form("survey_form"):
        st.markdown("#### A. 인구통계학적 정보")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_range = st.selectbox("연령대", ["10대 후반", "20대 초반", "20대 중후반", "30대 이상", "응답하지 않음"])
            gender = st.selectbox("성별", GENDER_OPTIONS)
            region = st.selectbox("주요 생활권", REGIONS)
        with c2:
            education = st.selectbox("교육 상태", EDUCATION_OPTIONS)
            subject_area = st.text_input("전공·관심 분야", "AI 서비스 기획")
            housing = st.selectbox("거주 형태", HOUSING_OPTIONS)
        with c3:
            economic_class = st.selectbox("주관적 경제적 위치", ECONOMIC_OPTIONS, index=2)
            future_focus = st.selectbox("연결하고 싶은 활동", ["취업", "창업", "연구·대학원", "지역문제 해결", "아직 미정"])

        st.markdown("#### B. 정치 관심과 정보 이용")
        c4, c5, c6 = st.columns(3)
        with c4:
            political_ideology = st.slider("정치 성향 자기배치: 0 진보 · 5 중도 · 10 보수", 0, 10, 5)
            political_interest = st.slider("정치·사회 현안에 대한 관심", 1, 5, 3)
        with c5:
            news_frequency = st.slider("뉴스와 시사 정보를 확인하는 빈도", 1, 5, 3)
            evidence_check = st.slider("주장이나 정보를 볼 때 출처를 확인하는 정도", 1, 5, 4)
        with c6:
            community_participation = st.slider("학내·지역·온라인 공론장 참여 경험", 1, 5, 2)
            public_discussion_comfort = st.slider("논쟁적인 이슈를 토론하는 데 느끼는 편안함", 1, 5, 3)
        information_sources = st.multiselect(
            "주로 신뢰하는 정보 출처",
            INFO_SOURCES,
            default=["논문·학술자료", "정부·공공기관 통계"],
        )

        st.markdown("#### C. 제도 신뢰")
        institutional_trust = {}
        trust_cols = st.columns(3)
        for idx, (key, label) in enumerate(TRUST_ITEMS):
            with trust_cols[idx % 3]:
                institutional_trust[key] = st.slider(label, 1, 5, 3, key=f"trust_{key}")

        st.markdown("#### D. 사회 현안 태도")
        issue_attitudes = {}
        issue_cols = st.columns(2)
        for idx, (key, label) in enumerate(ISSUE_ITEMS):
            with issue_cols[idx % 2]:
                issue_attitudes[key] = st.slider(label, 1, 5, 3, key=f"issue_{key}")

        open_issue_note = st.text_area(
            "최근 관심 있는 사회 현안과 그 이유",
            "AI 규제와 청년 고용 문제가 앞으로의 진로와 연결된다고 느낀다.",
            height=100,
        )
        consent = st.checkbox("수업 실습을 위해 입력값과 검증 기록을 로컬 파일에 저장하는 것에 동의합니다.")
        submitted = st.form_submit_button("페르소나 분석 시작")

    if submitted:
        if not valid_email(st.session_state.get("email", "")):
            st.error("이메일을 먼저 입력해야 저장할 수 있습니다.")
        elif not consent:
            st.error("저장 동의가 필요합니다.")
        else:
            form = {
                "age_range": age_range,
                "gender": gender,
                "region": region,
                "education": education,
                "subject_area": subject_area.strip() or "미입력",
                "housing": housing,
                "economic_class": economic_class,
                "future_focus": future_focus,
                "political_ideology": political_ideology,
                "political_interest": political_interest,
                "news_frequency": news_frequency,
                "evidence_check": evidence_check,
                "community_participation": community_participation,
                "public_discussion_comfort": public_discussion_comfort,
                "information_sources": information_sources,
                "institutional_trust": institutional_trust,
                "issue_attitudes": issue_attitudes,
                "open_issue_note": open_issue_note.strip(),
            }
            profile = build_profile(form, st.session_state["email"])
            st.session_state["profile"] = profile
            runtime = gemini_runtime_config()
            api_key = runtime["GEMINI_API_KEY"]
            model = runtime["GEMINI_MODEL"]

            if not api_key:
                st.error("분석 환경이 아직 준비되지 않았습니다. 관리자에게 Streamlit Secrets 설정을 확인해 달라고 요청하세요.")
                st.stop()

            with st.status("분석 중입니다. 입력한 사회조사 응답을 바탕으로 페르소나를 구성하고 있습니다.", expanded=True) as status:
                st.write("1단계: 입력값을 사회조사형 축으로 정리합니다.")
                st.write("2단계: 분석 엔진으로 결과 생성을 준비합니다.")
                try:
                    raw_text, raw_response = call_gemini(api_key, model, persona_prompt(profile))
                    llm_json = parse_json_from_text(raw_text)
                    if not llm_json:
                        raise ValueError("LLM 응답을 JSON으로 해석하지 못했습니다.")
                    persona = normalize_llm_persona(profile, llm_json, raw_text)
                    persona["raw_response"] = raw_response
                    st.write("3단계: LLM 분석 결과를 구조화했습니다.")
                    status.update(label="분석이 완료되었습니다.", state="complete")
                except Exception as exc:
                    status.update(label="분석에 실패했습니다.", state="error")
                    st.error("분석이 완료되지 않았습니다. 관리자에게 분석 설정을 확인해 달라고 요청하세요.")
                    st.session_state["last_analysis_error"] = str(exc)
                    st.stop()

            st.session_state["persona"] = persona
            append_jsonl(RECORDS_PATH, {"created_at": now_iso(), **persona})
            st.success("페르소나 분석이 저장되었습니다. 3번 탭에서 결과를 확인하세요.")

with tab_persona:
    st.subheader("페르소나 분석 결과")
    persona = st.session_state.get("persona")
    if not persona:
        st.info("2번 입력 탭에서 페르소나를 먼저 분석하세요.")
    else:
        render_persona(persona)
        st.warning("이 결과는 사회조사형 자기응답을 바탕으로 한 실습용 가설입니다. 실제 조사 결과나 정치 선택 예측으로 사용하지 않습니다.")
        st.download_button(
            "개인 분석 보고서 다운로드",
            data=persona_report(persona, st.session_state.get("last_issue_answer")),
            file_name=f"persona_report_{safe_filename_part(persona['profile'].get('email', 'user'))}.md",
            mime="text/markdown",
        )
        with st.expander("분석 원본 JSON"):
            st.json({k: v for k, v in persona.items() if k != "raw_response"})

with tab_issue:
    st.subheader("10개 현안 5점 응답 검증")
    persona = st.session_state.get("persona")
    if not persona:
        st.info("2번 입력 탭에서 페르소나를 먼저 분석하세요.")
    else:
        st.write("아래 10개 현안에 대해 페르소나가 1~5점 척도로 응답합니다. 이후 각 응답을 맞다/틀리다로 검증합니다.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"번호": idx + 1, "현안": issue_name, "질문": question}
                    for idx, (issue_name, question) in enumerate(CURRENT_ISSUES.items())
                ]
            ),
            width="stretch",
        )
        st.caption("척도: 1점 매우 반대 · 2점 반대 · 3점 중립/판단 유보 · 4점 찬성 · 5점 매우 찬성")
        runtime = gemini_runtime_config()
        api_key = runtime["GEMINI_API_KEY"]
        model = runtime["GEMINI_MODEL"]

        if st.button("10개 현안 응답 생성", type="primary"):
            if not api_key:
                st.error("현안 응답 분석 환경이 아직 준비되지 않았습니다. 관리자에게 Streamlit Secrets 설정을 확인해 달라고 요청하세요.")
                st.stop()
            with st.status("페르소나가 10개 현안에 대해 5점 척도로 응답하는 중입니다.", expanded=True) as status:
                st.write("페르소나 분석 결과와 10개 현안 목록을 결합합니다.")
                try:
                    raw_text, raw_response = call_gemini(api_key, model, issue_prompt(persona, CURRENT_ISSUES))
                    issue_json = parse_json_from_text(raw_text)
                    if not issue_json:
                        raise ValueError("LLM 응답을 JSON으로 해석하지 못했습니다.")
                    issue_json = normalize_issue_batch(issue_json)
                    issue_json["raw_response"] = raw_response
                    issue_json["generated_at"] = now_iso()
                    status.update(label="응답 생성이 완료되었습니다.", state="complete")
                except Exception as exc:
                    status.update(label="응답 생성에 실패했습니다.", state="error")
                    st.error("현안 응답이 완료되지 않았습니다. 관리자에게 분석 설정을 확인해 달라고 요청하세요.")
                    st.session_state["last_issue_error"] = str(exc)
                    st.stop()

            st.session_state["last_issue_answer"] = issue_json

        answer = st.session_state.get("last_issue_answer")
        if answer:
            responses = answer.get("responses", [])
            if answer.get("overall_pattern"):
                st.info(answer["overall_pattern"])

            response_df = pd.DataFrame(
                [
                    {
                        "번호": idx + 1,
                        "현안": item.get("issue_name", ""),
                        "질문": item.get("question", ""),
                        "페르소나 응답": item.get("label", ""),
                        "응답 문장": item.get("response_text", ""),
                        "이유": item.get("reason", ""),
                        "신뢰도": item.get("confidence", ""),
                        "불확실성": item.get("uncertainty", ""),
                    }
                    for idx, item in enumerate(responses)
                ]
            )
            st.markdown("#### 페르소나의 10개 현안 5점 응답")
            st.dataframe(response_df, width="stretch")

            with st.form("issue_evaluation_form"):
                st.markdown("#### 내 검증")
                st.caption("각 현안별로 페르소나 응답이 나와 맞으면 ‘맞다’, 다르면 ‘틀리다’를 선택합니다. 틀린 경우 실제 내 응답 점수와 정정 이유를 적습니다.")
                validation_inputs = []
                for idx, item in enumerate(responses):
                    st.markdown(f"**{idx + 1}. {item.get('issue_name', '')}**")
                    st.write(f"페르소나 응답: **{item.get('label', '')}**")
                    if item.get("response_text"):
                        st.write(item["response_text"])
                    st.caption(item.get("verification_hint", ""))
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        verdict = st.radio(
                            "검증",
                            ["맞다", "틀리다"],
                            horizontal=True,
                            key=f"verdict_{idx}",
                        )
                    with c2:
                        corrected_score = st.selectbox(
                            "틀렸다면 실제 내 응답",
                            options=list(LIKERT_LABELS.keys()),
                            format_func=lambda value: LIKERT_LABELS[value],
                            index=max(0, int(item.get("score", 3)) - 1),
                            key=f"corrected_score_{idx}",
                        )
                    correction_reason = st.text_area(
                        "틀린 이유 또는 정정 의견",
                        placeholder="예: 나는 정책 개입에는 찬성하지만 비용 부담에는 더 신중하다.",
                        key=f"correction_reason_{idx}",
                        height=74,
                    )
                    validation_inputs.append(
                        {
                            "item": item,
                            "verdict": verdict,
                            "corrected_score": corrected_score,
                            "correction_reason": correction_reason.strip(),
                        }
                    )
                saved = st.form_submit_button("10개 검증 기록 저장")
            if saved:
                missing = [
                    row["item"].get("issue_name", "")
                    for row in validation_inputs
                    if row["verdict"] == "틀리다" and not row["correction_reason"]
                ]
                if missing:
                    st.error("틀리다로 표시한 현안에는 정정 의견을 입력해야 합니다: " + ", ".join(missing))
                else:
                    batch_id = now_iso()
                    for row in validation_inputs:
                        item = row["item"]
                        append_csv(
                            EVALUATIONS_PATH,
                            {
                                "created_at": batch_id,
                                "email": persona["profile"].get("email", ""),
                                "persona_type": persona.get("persona_type", ""),
                                "issue_name": item.get("issue_name", ""),
                                "question": item.get("question", ""),
                                "predicted_score": item.get("score", ""),
                                "predicted_label": item.get("label", ""),
                                "predicted_response_text": item.get("response_text", ""),
                                "predicted_reason": item.get("reason", ""),
                                "predicted_confidence": item.get("confidence", ""),
                                "verdict": row["verdict"],
                                "corrected_score": row["corrected_score"] if row["verdict"] == "틀리다" else "",
                                "corrected_label": LIKERT_LABELS[row["corrected_score"]] if row["verdict"] == "틀리다" else "",
                                "correction_reason": row["correction_reason"],
                            },
                        )
                    st.success("10개 현안 검증 기록이 저장되었습니다.")

with tab_admin:
    st.subheader("관리자 대시보드")
    st.caption("분석 설정과 전체 응답 기록은 관리자만 확인합니다.")

    if not st.session_state.get("admin_authenticated", False):
        if not configured_admin_password():
            st.error("관리자 비밀번호가 설정되어 있지 않습니다. Streamlit Secrets 또는 환경변수에 ADMIN_PASSWORD를 먼저 등록하세요.")
            st.code('ADMIN_PASSWORD = "강사용_비밀번호"\nGEMINI_API_KEY = "YOUR_GEMINI_API_KEY"', language="toml")
        else:
            with st.form("admin_login_form"):
                password = st.text_input("관리자 비밀번호", type="password")
                login = st.form_submit_button("관리자 로그인")
            if login:
                if admin_login(password):
                    st.session_state["admin_authenticated"] = True
                    st.success("관리자 로그인이 완료되었습니다.")
                    st.rerun()
                else:
                    st.error("비밀번호가 맞지 않습니다.")

    if st.session_state.get("admin_authenticated", False):
        runtime = gemini_runtime_config()
        st.markdown("#### Gemini 서버 설정")
        c1, c2, c3 = st.columns(3)
        c1.metric("키 상태", gemini_admin_status_label())
        c2.metric("모델", runtime["GEMINI_MODEL"])
        c3.metric("운영 방식", "Secrets 기반")
        st.info(
            "배포 운영에서는 Streamlit Cloud Secrets에 GEMINI_API_KEY와 ADMIN_PASSWORD를 한 번만 등록합니다. "
            "사용자는 API 키를 입력하지 않으며, 관리자 대시보드에도 매번 키를 넣을 필요가 없습니다."
        )

        st.divider()
        records = load_jsonl(RECORDS_PATH)
        evals = load_evaluations(EVALUATIONS_PATH)

        m1, m2, m3 = st.columns(3)
        m1.metric("페르소나 기록", len(records))
        m2.metric("검증 기록", 0 if evals.empty else len(evals))
        m3.metric("저장 위치", "로컬 data/")

        with st.expander("테스트 데이터 삭제", expanded=False):
            st.warning("삭제한 기록은 복구할 수 없습니다. 테스트 기록을 정리할 때만 사용하세요.")
            email_values = record_email_values(records, evals)
            if email_values:
                with st.form("delete_email_data_form"):
                    selected_email = st.selectbox("삭제할 이메일 또는 기존 식별자", email_values)
                    confirm_delete = st.text_input("선택 기록 삭제 확인 문구", placeholder="DELETE")
                    delete_selected = st.form_submit_button("선택한 사용자 기록 삭제")
                if delete_selected:
                    if confirm_delete.strip() != "DELETE":
                        st.error("삭제하려면 확인 문구에 DELETE를 입력하세요.")
                    else:
                        deleted_records, deleted_evals = delete_data_for_email(selected_email)
                        st.success(f"{selected_email} 기록을 삭제했습니다. 페르소나 {deleted_records}건, 검증 {deleted_evals}건")
                        st.rerun()
            else:
                st.info("삭제할 사용자 기록이 없습니다.")

            with st.form("delete_all_data_form"):
                confirm_all = st.text_input("전체 테스트 데이터 삭제 확인 문구", placeholder="DELETE ALL")
                delete_all = st.form_submit_button("전체 테스트 데이터 삭제")
            if delete_all:
                if confirm_all.strip() != "DELETE ALL":
                    st.error("전체 삭제하려면 확인 문구에 DELETE ALL을 입력하세요.")
                else:
                    deleted_records, deleted_evals = delete_all_data()
                    st.success(f"전체 테스트 데이터를 삭제했습니다. 페르소나 {deleted_records}건, 검증 {deleted_evals}건")
                    st.rerun()

        if records:
            score_df = records_to_frame(records)
            numeric_cols = [col for col in score_df.columns if pd.api.types.is_numeric_dtype(score_df[col])]
            st.markdown("#### 분석 축 분포")
            if numeric_cols:
                st.bar_chart(score_df[numeric_cols])
            st.dataframe(score_df, width="stretch")
            st.download_button(
                "페르소나 기록 CSV 다운로드",
                data=score_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="persona_records_export.csv",
                mime="text/csv",
            )
        else:
            st.info("아직 저장된 페르소나 기록이 없습니다.")

        if not evals.empty:
            st.markdown("#### 검증 판정 분포")
            st.bar_chart(evals["verdict"].value_counts())
            st.dataframe(evals.tail(30), width="stretch")
            st.download_button(
                "검증 기록 CSV 다운로드",
                data=evals.to_csv(index=False).encode("utf-8-sig"),
                file_name="persona_evaluations_export.csv",
                mime="text/csv",
            )
        else:
            st.info("아직 저장된 검증 기록이 없습니다.")
