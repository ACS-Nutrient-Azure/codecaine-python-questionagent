"""
question_agent.py

LangChain AgentExecutor 기반 Question Agent 핵심 로직.

흐름:
  1. Supervisor로부터 요청 수신
  2. LLM(Claude 3.5 Sonnet)이 질문을 보고 필요한 tool을 선택해 실행
  3. tool 결과를 바탕으로 한국어 답변 생성
  4. 사용된 tool 목록(sources_used)과 함께 응답 반환

사용 가능한 Tools:
  - search_knowledge_base : ChromaDB에서 영양소/의약품 전문 지식 검색
  - search_web            : DuckDuckGo로 최신 건강 정보 웹 검색
  - get_user_data         : chatbot_userdata 테이블 조회 (신체 정보, 알레르기 등)
  - get_supplements       : chatbot_supplements 테이블 조회 (복용 중인 영양제)
  - get_analysis_result   : chatbot_analysis_result 테이블 조회 (분석 결과, 추천)
"""
import json
import logging

from app.core.config import settings
from app.schemas.agent import QuestionRequest, QuestionResponse
from app.services import db_tools, kb_retriever

logger = logging.getLogger(__name__)

# LLM에게 역할과 사용 가능한 tool을 안내하는 시스템 프롬프트
SYSTEM_PROMPT = """당신은 영양제 및 건강 관련 질문에 답변하는 전문 AI 어시스턴트입니다.
사용자의 질문에 답하기 위해 필요한 경우 아래 도구를 활용하세요.

- search_knowledge_base: 영양소/의약품 상호작용 등 전문 지식 검색
- search_web: KB나 DB로 답할 수 없는 최신 정보 검색
- get_user_data: 사용자 기본 신체 정보, 알레르기, 만성질환 조회
- get_supplements: 사용자가 복용 중인 영양제 조회
- get_analysis_result: 사용자의 영양소 분석 결과 및 추천 조회

반드시 한국어로 답변하세요."""


class QuestionAgent:
    def __init__(self):
        import boto3
        from langchain_aws import ChatBedrock

        if settings.USE_LOCAL_TEST:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
            )
        else:
            session = boto3.Session(region_name=settings.AWS_REGION)
            self.llm = ChatBedrock(
                client=session.client("bedrock-runtime"),
                model_id=settings.BEDROCK_MODEL_ID,
            )

        self.memory_client = None
        if settings.USE_MEMORY:
            try:
                from bedrock_agentcore.memory import MemoryClient
                self.memory_client = MemoryClient(region_name=settings.AWS_REGION)
                logger.info("[QuestionAgent] MemoryClient initialized.")
            except ModuleNotFoundError:
                logger.warning("[QuestionAgent] bedrock_agentcore.memory not available, memory disabled.")
            except Exception as e:
                logger.error(f"[QuestionAgent] MemoryClient init failed, memory disabled: {e}", exc_info=True)

    def _build_tools(self, cognito_id: str) -> list:
        from langchain.tools import StructuredTool
        from duckduckgo_search import DDGS

        def search_web_ddg(query: str) -> str:
            try:
                results = DDGS().text(query, max_results=3)
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return f"검색 실패: {str(e)}"

        async def _get_user_data(query: str = "") -> str:
            return await db_tools.get_user_data(cognito_id)

        async def _get_supplements(query: str = "") -> str:
            return await db_tools.get_supplements(cognito_id)

        async def _get_analysis_result(query: str = "") -> str:
            return await db_tools.get_analysis_result(cognito_id)

        return [
            StructuredTool.from_function(
                name="search_knowledge_base",
                func=kb_retriever.retrieve,
                description="영양소, 의약품 상호작용 등 전문 지식을 Knowledge Base에서 검색합니다. 입력: 검색 쿼리 문자열",
            ),
            StructuredTool.from_function(
                name="search_web",
                func=search_web_ddg,
                description="최신 건강 정보나 일반 지식을 웹에서 검색합니다. 입력: 검색 쿼리 문자열",
            ),
            StructuredTool.from_function(
                name="get_user_data",
                coroutine=_get_user_data,
                description="사용자의 기본 신체 정보(키, 몸무게, 알레르기, 만성질환 등)를 조회합니다. 입력: 무시됨",
            ),
            StructuredTool.from_function(
                name="get_supplements",
                coroutine=_get_supplements,
                description="사용자가 현재 복용 중인 영양제 목록과 성분을 조회합니다. 입력: 무시됨",
            ),
            StructuredTool.from_function(
                name="get_analysis_result",
                coroutine=_get_analysis_result,
                description="사용자의 영양소 분석 결과, 영양소 갭, 추천 영양제를 조회합니다. 입력: 무시됨",
            ),
        ]

    async def run(self, req: QuestionRequest) -> QuestionResponse:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        print(f"\n[QUESTION AGENT] Received request from {req.cognito_id}")
        print(f"  - chat_history: {req.chat_history[:100]}...")
        
        # 1. AgentCore Memory에서 이전 대화 불러오기
        memory_context = ""
        if self.memory_client and settings.MEMORY_ID:
            try:
                resp = self.memory_client.list_events(
                    memoryId=settings.MEMORY_ID,
                    actorId=req.cognito_id,
                    sessionId=str(req.chat_result_id),
                    maxResults=10
                )
                conversations = resp.get("events", [])
                if conversations:
                    memory_context = "\n\n[이전 대화 맥락]\n" + self._format_memory(conversations)
                    print(f"[QUESTION AGENT] Loaded {len(conversations)} previous messages from memory")
            except Exception as e:
                print(f"[QUESTION AGENT] Memory load failed: {e}")
        
        tools = self._build_tools(req.cognito_id)

        # system 프롬프트 + 사용자 입력 + agent_scratchpad(tool 실행 중간 결과) 구조
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),  # LangChain이 tool 호출 결과를 여기에 채움
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=True, return_intermediate_steps=True)

        user_input = self._build_input(req) + memory_context
        result = await executor.ainvoke({"input": user_input})

        sources_used = self._extract_sources(result)

        # langchain 0.3.x에서 output이 content block 리스트로 올 수 있음
        output = result["output"]
        if isinstance(output, list):
            output = " ".join(
                block.get("text", "") for block in output
                if isinstance(block, dict) and block.get("type") == "text"
            )

        print(f"[QUESTION AGENT] Answer: {str(output)[:100]}...")
        print(f"[QUESTION AGENT] Sources used: {sources_used}")
        
        # 2. 대화를 AgentCore Memory에 저장
        if self.memory_client and settings.MEMORY_ID:
            try:
                self.memory_client.create_event(
                    memoryId=settings.MEMORY_ID,
                    actorId=req.cognito_id,
                    sessionId=str(req.chat_result_id),
                    messages=[
                        {"role": "USER", "content": req.chat_history},
                        {"role": "ASSISTANT", "content": output}
                    ]
                )
                print(f"[QUESTION AGENT] Saved conversation to memory")
            except Exception as e:
                print(f"[QUESTION AGENT] Memory save failed: {e}")

        return QuestionResponse(
            cognito_id=req.cognito_id,
            answer=output,
            sources_used=sources_used,
        )

    def _format_memory(self, conversations: list) -> str:
        """AgentCore Memory 대화를 텍스트로 변환"""
        lines = []
        for conv in conversations:
            messages = conv.get("messages", [])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "USER":
                    lines.append(f"사용자: {content}")
                elif role == "ASSISTANT":
                    lines.append(f"AI: {content}")
        return "\n".join(lines[-20:])  # 최근 20개만

    def _build_input(self, req: QuestionRequest) -> str:
        """
        LLM에 전달할 입력 문자열 구성.
        채팅 내용 외에 건강검진/의약품 데이터가 있으면 함께 포함.
        """
        parts = [f"사용자 질문/대화:\n{req.chat_history}"]
        if req.codef_health_data:
            parts.append("건강검진 데이터:\n" + json.dumps(req.codef_health_data, ensure_ascii=False))
        if req.codef_medication_info:
            parts.append("복용 의약품:\n" + json.dumps(req.codef_medication_info, ensure_ascii=False))
        return "\n\n".join(parts)

    def _extract_sources(self, result: dict) -> list[str]:
        """
        AgentExecutor의 intermediate_steps를 순회해
        실제로 사용된 tool 종류를 ["kb", "web", "db"] 형태로 추출.
        """
        sources = []
        for step in result.get("intermediate_steps", []):
            tool_name = step[0].tool if hasattr(step[0], "tool") else ""
            if "knowledge_base" in tool_name and "kb" not in sources:
                sources.append("kb")
            elif "web" in tool_name and "web" not in sources:
                sources.append("web")
            elif tool_name in ("get_user_data", "get_supplements", "get_analysis_result") and "db" not in sources:
                sources.append("db")
        return sources
