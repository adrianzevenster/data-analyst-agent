from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {
        "name": "Data Analyst Agent",
        "status": "ok",
        "endpoints": ["/health", "/uploads", "/datasets", "/chat"],
    }
