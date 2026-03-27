from fastapi import APIRouter, HTTPException
from app.schemas.agent import QuestionRequest, QuestionResponse
from app.services.question_agent import QuestionAgent

router = APIRouter()


@router.post("/invocations", response_model=QuestionResponse)
async def invocations(req: QuestionRequest):
    try:
        agent = QuestionAgent()
        return await agent.run(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
