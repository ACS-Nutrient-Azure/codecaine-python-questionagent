import json
import logging
import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.agent import QuestionRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/invocations")
async def invocations(request: Request):
    raw = await request.body()
    logger.info(f"[INVOCATIONS] raw body: {raw[:500]}")
    logger.info(f"[INVOCATIONS] content-type: {request.headers.get('content-type')}")

    try:
        data = json.loads(raw)
        req = QuestionRequest(**data)
    except Exception as e:
        logger.error(f"[INVOCATIONS] Request parse failed: {e} | raw={raw[:500]}")
        return JSONResponse(status_code=422, content={"error": f"Request parse error: {e}"})

    agent = request.app.state.agent
    if agent is None:
        logger.error("[INVOCATIONS] Agent not initialized")
        return JSONResponse(status_code=500, content={"error": "Agent initialization failed. Check startup logs."})

    try:
        result = await agent.run(req)
        # AgentCore가 파싱할 수 있도록 명시적 JSONResponse 반환
        return JSONResponse(content={
            "cognito_id": result.cognito_id,
            "answer": result.answer,
            "sources_used": result.sources_used,
        })
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"[INVOCATIONS] Agent run failed: {error_detail}")
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {str(e)}"})
