from fastapi import APIRouter

router = APIRouter()

@router.post("/message")
def send_message(message: str):
    """
    (Placeholder) Send a message to the matchmaker bot.
    """
    return {"response": f"Echo: {message}", "status": "processing"}
