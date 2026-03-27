# codecaine-python-chatbotagent (Question Agent)

영양제/건강 관련 일반 질문에 답변하는 에이전트.
LangChain AgentExecutor 기반으로 LLM이 필요한 tool을 스스로 선택해 실행.

## 역할

Supervisor Agent로부터 요청을 받아 채팅 내용을 분석하고, 필요한 tool을 선택해 실행한 뒤 한국어 답변을 반환.

## 전체 흐름

```
Supervisor Agent
    └─→ POST /invocations
            └─→ QuestionAgent.run()
                    └─→ LangChain AgentExecutor
                            ├─→ [필요 시] search_knowledge_base (ChromaDB)
                            ├─→ [필요 시] search_web (Tavily)
                            ├─→ [필요 시] get_user_data (PostgreSQL)
                            ├─→ [필요 시] get_supplements (PostgreSQL)
                            └─→ [필요 시] get_analysis_result (PostgreSQL)
                    └─→ 한국어 답변 반환
```

## 프로젝트 구조

```
codecaine-python-chatbotagent/
├── Dockerfile              # linux/arm64, Python 3.12, port 8080
├── requirements.txt
├── env.example             # 환경변수 예시 (.env로 복사 후 값 채우기)
├── kb_vector_db/           # ChromaDB Knowledge Base (analysisagent lpi_vector_db 복사본)
│                           # 컨테이너 이미지에 포함되어 배포됨
└── app/
    ├── main.py             # FastAPI 앱, /ping + /invocations 라우터 등록
    ├── core/config.py      # pydantic-settings 기반 환경변수 관리
    ├── schemas/agent.py    # QuestionRequest / QuestionResponse Pydantic 모델
    ├── api/routes/
    │   └── invocations.py  # POST /invocations 엔드포인트 (AgentCore 런타임 계약)
    └── services/
        ├── question_agent.py  # LangChain AgentExecutor + Tool 정의 (핵심 로직)
        ├── kb_retriever.py    # ChromaDB 검색 (Knowledge Base tool)
        └── db_tools.py        # PostgreSQL 조회 tool 3종
```

## Tools

| Tool | 설명 | 데이터 소스 |
|------|------|------------|
| `search_knowledge_base` | 영양소/의약품 상호작용 전문 지식 검색 | ChromaDB (kb_vector_db/) |
| `search_web` | 최신 건강 정보 웹 검색 | Tavily API |
| `get_user_data` | 사용자 기본 신체 정보, 알레르기, 만성질환 | chatbot_userdata |
| `get_supplements` | 복용 중인 영양제 목록 및 성분 | chatbot_supplements + chatbot_current_ingredients |
| `get_analysis_result` | 영양소 분석 결과, 갭, 추천 | chatbot_analysis_result + chatbot_nutrient_gap + chatbot_recommendations |

## API

### `POST /invocations`

**Request**
```json
{
  "cognito_id": "string",
  "chat_result_id": 123,
  "codef_health_data": {},
  "codef_medication_info": [],
  "chat_history": "사용자 질문 내용"
}
```

**Response**
```json
{
  "cognito_id": "string",
  "answer": "한국어 답변",
  "sources_used": ["kb", "web", "db"]
}
```

### `GET /ping`
AgentCore 헬스체크 엔드포인트. `{"status": "ok"}` 반환.

## 환경변수

| 변수 | 설명 |
|------|------|
| `AWS_REGION` | AWS 리전 (기본값: ap-northeast-2) |
| `BEDROCK_MODEL_ID` | Bedrock 모델 ID (기본값: claude-3-5-sonnet) |
| `KB_LOCAL_PATH` | ChromaDB 경로 (기본값: /app/kb_vector_db) |
| `KB_COLLECTION_NAME` | ChromaDB 컬렉션명 (기본값: lpi_interactions) |
| `TAVILY_API_KEY` | Tavily 검색 API 키 |
| `DB_HOST` | PostgreSQL 호스트 |
| `DB_PORT` | PostgreSQL 포트 (기본값: 5432) |
| `DB_NAME` | DB명 |
| `DB_USER` | DB 사용자 |
| `DB_PASSWORD` | DB 비밀번호 |

## 배포 주의사항

- **VPC 설정 필요**: AgentCore 컨테이너가 RDS에 접근하려면 배포 시 VPC 서브넷/보안그룹 설정 필요
- **IAM Role**: AgentCore 환경에서 Bedrock 호출은 IAM Role로 자동 인증됨
- **KB 포함**: `kb_vector_db/`는 Dockerfile에서 이미지에 포함됨
