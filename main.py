# main.py
import os
import uuid
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")  # اختياري لحماية البروكسي

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="JanitorAI -> Gemini Proxy")

class ChatRequest(BaseModel):
    model: Optional[str] = "gemini-1.0"
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    class Config:
        extra = "allow"

def extract_messages(openai_messages: List[Dict[str, Any]]):
    msgs = []
    for m in openai_messages:
        role = m.get("role", "user")
        content = None
        # Handle common shapes: content as string, parts list, or nested dicts
        if isinstance(m.get("content"), str):
            content = m.get("content")
        elif isinstance(m.get("content"), dict):
            content = m["content"].get("text") or m["content"].get("content")
        elif isinstance(m.get("parts"), list):
            content = "\n".join(m["parts"])
        else:
            content = m.get("text") or m.get("body") or ""
        msgs.append({"role": role, "content": content})
    return msgs

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    # تحقق مفتاح البروكسي إذا تم تعيينه
    if PROXY_API_KEY:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.split()[-1]
        if token != PROXY_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid proxy API key")

    body = await request.json()
    req = ChatRequest(**body)

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on proxy")

    messages = extract_messages(req.messages)

    try:
        # إعداد ThinkingConfig مع فرض thinking_level="high"
        thinking_cfg = genai.types.ThinkingConfig(thinking_level="high")

        # تهيئة أداة بحث جوجل
        google_tool = genai.types.Tool(google_search=genai.types.GoogleSearch())

        # تحضير رسائل بصيغة متوافقة مع مكتبة genai
        g_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        # استدعاء Gemini (واجهة المكتبة قد تختلف بإصدار الحزمة)
        response = genai.chat.completions.create(
            model=req.model or "gemini-1.0",
            messages=g_messages,
            thinking_config=thinking_cfg,
            tools=[google_tool],
            max_output_tokens=req.max_tokens,
            temperature=req.temperature
        )
    except Exception as e:
        # إرجاع تفاصيل الخطأ لتسهيل تصحيح توافق واجهة المكتبة
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

    # محاولة استخراج نص من استجابة Gemini بطرائق شائعة
    text_out = None
    try:
        # شائع: response.choices[0].message.content أو response.output[0].content
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice0 = response.choices[0]
            msg = getattr(choice0, "message", None)
            if msg:
                # msg قد يكون dict أو كائن
                if isinstance(msg, dict):
                    text_out = msg.get("content") or msg.get("text")
                else:
                    text_out = getattr(msg, "content", None) or getattr(msg, "text", None)
            else:
                text_out = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)

        if not text_out and hasattr(response, "output") and len(response.output) > 0:
            out0 = response.output[0]
            if isinstance(out0, dict):
                text_out = out0.get("content") or out0.get("text")
            else:
                text_out = getattr(out0, "content", None) or getattr(out0, "text", None)
    except Exception:
        text_out = None

    if not text_out:
        text_out = str(response)

    proxy_id = f"proxy-{uuid.uuid4()}"
    now_ts = int(time.time())
    openai_resp = {
        "id": proxy_id,
        "object": "chat.completion",
        "created": now_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_out
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {}
    }

    return openai_resp

@app.get("/health")
def health():
    return {"status": "ok"}
