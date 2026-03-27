from fastapi import FastAPI
from app.api.routes import invocations

app = FastAPI(title="Question Agent", version="1.0.0")
app.include_router(invocations.router)


@app.get("/ping")
async def ping():
    return {"status": "ok"}
