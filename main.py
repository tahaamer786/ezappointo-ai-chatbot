"""
Local test harness for Chatbot_FS router.
Run: uvicorn main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from fastapi import FastAPI
from Chatbot_FS import router

app = FastAPI(title="EZAppointo Chatbot - Local Test")
app.include_router(router)
