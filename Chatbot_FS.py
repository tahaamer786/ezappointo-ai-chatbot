import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL_DIR = "/app/models/all-MiniLM-L6-v2"
_DEFAULT_CHATBOT_DATA_DIR = "/app/uploads/chatbot"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _first_writable_dir(*candidates: str) -> str:
    for path in candidates:
        if not path:
            continue
        try:
            return _ensure_dir(path)
        except OSError:
            continue
    return _ensure_dir(os.path.join(os.environ.get("TMPDIR", "/tmp"), "ezappointo-chatbot"))


def _configure_hf_cache() -> None:
    """Use a writable cache only when Hub download is needed (local dev)."""
    cache_root = os.environ.get("HF_CACHE_DIR")
    if not cache_root:
        cache_root = _first_writable_dir(
            "/app/uploads/.cache/huggingface",
            os.path.join(os.environ.get("TMPDIR", "/tmp"), "ezappointo-hf-cache"),
        )
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_root, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_root)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(cache_root, "sentence_transformers"))


def _resolve_paths() -> tuple[str, str]:
    model_path = os.environ.get("SENTENCE_TRANSFORMER_MODEL_PATH", _DEFAULT_MODEL_DIR)
    qdrant_path = os.path.join(
        _first_writable_dir(
            os.environ.get("CHATBOT_DATA_DIR", ""),
            _DEFAULT_CHATBOT_DATA_DIR,
            os.path.join(os.environ.get("TMPDIR", "/tmp"), "ezappointo-chatbot"),
        ),
        "qdrant_storage",
    )
    return model_path, qdrant_path


# Avoid defaulting HOME to /app on read-only images (HF would use /app/.cache).
if os.environ.get("HOME") in (None, "", "/nonexistent"):
    os.environ["HOME"] = os.environ.get("APP_HOME", os.environ.get("TMPDIR", "/tmp"))

_MODEL_PATH, QDRANT_PATH = _resolve_paths()

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid

# ── Router (prefix = /api/v1/chatbot) ────────────────────────────────────────
router = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["chatbot"]
)

# ── Configurations ────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Baked into the Docker image at build time; Hub + cache only for local dev without that path.
if os.path.isdir(_MODEL_PATH):
    embedder = SentenceTransformer(_MODEL_PATH)
else:
    _configure_hf_cache()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

q_client = QdrantClient(path=QDRANT_PATH)
COLLECTION_NAME = "ez_knowledge_base"

# Create collection only if it doesn't exist yet
if not q_client.collection_exists(COLLECTION_NAME):
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        ),
    )

sessions_db = {}
tickets_db = []


# ── Pydantic Models ───────────────────────────────────────────────────────────
class UserIn(BaseModel):
    name: str
    email: str


class ChatMsg(BaseModel):
    session_id: str
    user_message: str


# ── Helper Functions ──────────────────────────────────────────────────────────
def get_embedding(text: str):
    return embedder.encode(text).tolist()


def search_knowledge(query_text: str):
    query_vector = get_embedding(query_text)
    search_result = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    )
    context = "\n".join([res.payload["info"] for res in search_result.points])
    return context


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/add-to-knowledge-base")
async def add_knowledge(text: str, metadata: dict = {}):
    vector = get_embedding(text)
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"info": text, **metadata}
            )
        ]
    )
    return {"status": "Added to Knowledge Base Successfully"}


@router.post("/start-session")
async def start_session(user: UserIn):
    session_id = f"session_{user.email}"
    system_instruction = (
        f"You are the EZappointo AI assistant. Patient Name: {user.name}. "
        "Your goal is to assist with doctor appointments using the provided context. "
        "Be professional and brief. If the context doesn't have the answer, ask for manual review."
    )
    sessions_db[session_id] = {
        "details": user.dict(),
        "history": [{"role": "system", "content": system_instruction}]
    }
    return {"session_id": session_id, "status": "Ready to Chat"}


@router.post("/get-response")
async def get_ai_response(chat: ChatMsg):
    if chat.session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions_db[chat.session_id]
    context = search_knowledge(chat.user_message)
    enriched_prompt = f"Context from Knowledge Base:\n{context}\n\nUser Question: {chat.user_message}"
    session["history"].append({"role": "user", "content": enriched_prompt})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=session["history"]
        )
        ai_reply = response.choices[0].message.content
        session["history"][-1] = {"role": "user", "content": chat.user_message}
        session["history"].append({"role": "assistant", "content": ai_reply})
        return {"reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-ticket")
async def create_ticket(session_id: str):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    user_info = sessions_db[session_id]["details"]
    ticket = {
        "ticket_id": f"TKT-{datetime.now().strftime('%Y%m%d%H%M')}",
        "patient": user_info["name"],
        "email": user_info["email"],
        "status": "Pending/Manual Review",
        "timestamp": str(datetime.now())
    }
    tickets_db.append(ticket)
    return {"status": "Ticket Created Successfully", "ticket_details": ticket}