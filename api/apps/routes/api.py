from fastapi import APIRouter


router = APIRouter(tags=["api"], prefix="/api")

@router.get("/healthz")
async def healthz_handler():
    return {"status": "ok"}