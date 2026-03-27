from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AWS_REGION: str = "ap-northeast-2"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Knowledge Base
    KB_LOCAL_PATH: str = "/app/kb_vector_db"
    KB_COLLECTION_NAME: str = "lpi_interactions"
    KB_TOP_K: int = 3

    # PostgreSQL
    DB_HOST: str = ""
    DB_PORT: int = 5432
    DB_NAME: str = ""
    DB_USER: str = ""
    DB_PASSWORD: str = ""

    # 로컬 테스트용
    USE_LOCAL_TEST: bool = False
    # GEMINI_API_KEY: str = ""
    # GEMINI_MODEL_ID: str = "gemini-2.0-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    DB_API_URL: str = "http://localhost:8006"

    # AgentCore Memory
    USE_MEMORY: bool = False
    MEMORY_ID: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
