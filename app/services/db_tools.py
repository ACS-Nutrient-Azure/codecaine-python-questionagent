"""
db_tools.py

PostgreSQL(RDS) 직접 연결 기반 DB 조회 tool 모음.
LangChain Tool로 등록되어 LLM이 필요 시 호출.

테이블 구조 (reference/chatbot_db.sql 참고):
  - chatbot_userdata           : 사용자 기본 정보 (신체, 알레르기, 만성질환)
  - chatbot_supplements        : 복용 중인 영양제
  - chatbot_current_ingredients: 영양제별 성분
  - chatbot_analysis_result    : 영양소 분석 결과 요약
  - chatbot_nutrient_gap       : 영양소별 갭(부족량)
  - chatbot_recommendations    : 추천 영양제

주의: AgentCore 컨테이너가 RDS에 접근하려면
      배포 시 VPC 서브넷/보안그룹 설정이 필요함.
"""
import json
import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)


async def _fetch(query: str, *args) -> list[dict]:
    """
    단일 쿼리를 실행하고 결과를 dict 리스트로 반환하는 내부 헬퍼.
    로컬 테스트 모드: Mock DB API 호출
    실제 배포: PostgreSQL 직접 연결
    """
    # 로컬 테스트 모드: HTTP API 호출
    if settings.USE_LOCAL_TEST:
        import httpx
        # 쿼리 대신 cognito_id만 추출 (간단한 파싱)
        cognito_id = args[0] if args else ""
        
        # 쿼리 종류 판단
        if "chatbot_userdata" in query:
            url = f"{settings.DB_API_URL}/api/db/userdata/{cognito_id}"
        elif "chatbot_supplements" in query:
            url = f"{settings.DB_API_URL}/api/db/supplements/{cognito_id}"
        elif "chatbot_analysis_result" in query:
            url = f"{settings.DB_API_URL}/api/db/analysis/{cognito_id}"
        else:
            return []
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return [data] if isinstance(data, dict) else data
    
    # 실제 배포: PostgreSQL 연결
    conn = await asyncpg.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
    )
    try:
        rows = await conn.fetch(query, *args)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def get_user_data(cognito_id: str) -> str:
    """
    chatbot_userdata 테이블에서 사용자 기본 정보 조회.
    반환: 생년월일, 성별, 키, 몸무게, 알레르기, 만성질환, 현재 상태 등
    """
    rows = await _fetch("SELECT * FROM chatbot_userdata WHERE cognito_id = $1", cognito_id)
    return json.dumps(rows, ensure_ascii=False, default=str) if rows else "데이터 없음"


async def get_supplements(cognito_id: str) -> str:
    """
    chatbot_supplements + chatbot_current_ingredients JOIN 조회.
    복용 중인 영양제 목록과 각 영양제의 성분/함량을 함께 반환.
    chat_is_active 필터 없이 전체 이력 반환 (LLM이 판단하도록).
    """
    rows = await _fetch(
        """
        SELECT s.*, ci.chat_ingredient_name, ci.chat_nutrient_amount
        FROM chatbot_supplements s
        LEFT JOIN chatbot_current_ingredients ci ON s.chat_current_id = ci.chat_current_id
        WHERE s.cognito_id = $1
        """,
        cognito_id,
    )
    return json.dumps(rows, ensure_ascii=False, default=str) if rows else "데이터 없음"


async def get_analysis_result(cognito_id: str) -> str:
    """
    chatbot_analysis_result + chatbot_nutrient_gap + chatbot_recommendations 3-way JOIN 조회.
    가장 최근 분석 결과부터 반환 (ORDER BY created_at DESC).
    반환: 분석 요약, 영양소별 현재 섭취량/갭, 추천 영양제 순위
    """
    rows = await _fetch(
        """
        SELECT ar.chat_result_id, ar.chat_summary,
               ng.chat_current_amount, ng.chat_gap_amount, ng.chat_unit,
               rec.chat_recommend_serving, rec.chat_rank
        FROM chatbot_analysis_result ar
        LEFT JOIN chatbot_nutrient_gap ng ON ar.chat_result_id = ng.chat_result_id
        LEFT JOIN chatbot_recommendations rec ON ng.chat_gap_id = rec.chat_gap_id
        WHERE ar.cognito_id = $1
        ORDER BY ar.created_at DESC
        """,
        cognito_id,
    )
    return json.dumps(rows, ensure_ascii=False, default=str) if rows else "데이터 없음"
