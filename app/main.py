import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.routes import invocations
from app.telemetry import setup_xray, XRayMiddleware
from app.services import kb_retriever
from app.services.question_agent import QuestionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    setup_xray("cdci-prd-question-agent")
except Exception as e:
    logging.getLogger(__name__).warning("X-Ray setup failed (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()

    # KB 로드
    try:
        logger.info("[STARTUP] Warming up KB...")
        await loop.run_in_executor(None, kb_retriever._get_kb)
        logger.info("[STARTUP] KB loaded.")
    except Exception as e:
        logger.error(f"[STARTUP] KB load failed: {e}", exc_info=True)
        # KB 실패해도 retrieve tool 내부에서 예외 처리되므로 계속 진행

    # Agent 초기화 — lifespan에서 완료 후 서버 요청 수락
    try:
        logger.info("[STARTUP] Warming up QuestionAgent...")
        agent = await loop.run_in_executor(None, QuestionAgent)
        app.state.agent = agent
        app.state.ready = True
        logger.info("[STARTUP] Ready.")
    except Exception as e:
        logger.error(f"[STARTUP] QuestionAgent init failed: {e}", exc_info=True)
        app.state.agent = None
        app.state.ready = False

    yield


app = FastAPI(title="Question Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(XRayMiddleware)
app.include_router(invocations.router)


@app.get("/ping")
async def ping():
    status = "Healthy" if app.state.ready else "HealthyBusy"
    return {"status": status, "time_of_last_update": int(time.time())}
