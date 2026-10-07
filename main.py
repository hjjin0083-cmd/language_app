import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="4개 국어 대화형 학습기")

# API 키 설정 (Windows 환경변수에서 가져옴)
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key) if api_key else None


class LanguageDetail(BaseModel):
    text: str = Field(description="해당 언어로 번역/응답된 문장")
    structure: str = Field(description="어순 구조 (예: SOV, SVO 등)")
    pronunciation: Optional[str] = Field(
        default=None,
        description="발음 안내. 중국어는 병음/성조, 일본어는 히라가나/로마자 포함",
    )


class LanguagesDict(BaseModel):
    korean: LanguageDetail
    english: LanguageDetail
    chinese: LanguageDetail
    japanese: LanguageDetail


class MetadataDetail(BaseModel):
    contrast_note: str = Field(
        description="어순, 핵심 문법 및 대화 맥락에 대한 한국어 설명/답변"
    )


class MultiLingualResponse(BaseModel):
    original_input: str
    languages: LanguagesDict
    metadata: MetadataDetail


class ChatMessage(BaseModel):
    role: str  # 'user' 또는 'model'
    content: str


class UserInput(BaseModel):
    text: str
    history: Optional[List[ChatMessage]] = []


@app.get("/")
async def get_ui():
    return FileResponse(BASE_DIR / "index.html")


def call_gemini_api(prompt_text: str, history: List[ChatMessage]) -> str:
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. Windows 환경변수에 API 키를 설정해주세요."
        )

    # 최근 6개 메시지만 포함하여 속도 최적화
    recent_history = history[-6:] if history else []

    history_str = ""
    if recent_history:
        history_str = "\n[이전 대화 내역]\n"
        for msg in recent_history:
            role_name = "사용자" if msg.role == "user" else "AI 대답"
            history_str += f"{role_name}: {msg.content}\n"

    prompt = f"""
당신은 4개 국어(한국어, 영어, 중국어, 일본어) 동시 학습을 돕는 AI 튜터입니다.
이전 대화 내역이 있다면 맥락을 반영하여 질문에 응답하거나 문장을 번역/수정하세요.

{history_str}
[현재 사용자 입력]
"{prompt_text}"

[작성 규칙]
1. languages 속성에는 korean, english, chinese, japanese 키를 반드시 포함할 것.
2. pronunciation에는 각 언어별 발음 안내를 작성할 것.
   - 중국어: 한자와 병음, 성조 포함
   - 일본어: 자연스러운 표기와 히라가나 발음 포함
3. metadata.contrast_note에는 어순 차이(SOV/SVO), 어휘/문법 설명, 또는 사용자의 질의에 대한 자연스러운 대화형 답변을 한국어로 작성할 것.
4. 사용자가 특정 언어나 문장 수정을 요구할 경우, 전체 4개 국어 카드에도 변경사항을 반영하여 출력할 것.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MultiLingualResponse,
            temperature=0.4,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini API가 빈 응답을 반환했습니다.")

    return response.text


@app.post("/api/chat", response_model=MultiLingualResponse)
async def chat_endpoint(payload: UserInput):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="문장이나 질문을 입력해주세요.")

    try:
        raw_response = await asyncio.wait_for(
            asyncio.to_thread(call_gemini_api, text, payload.history or []),
            timeout=60.0,
        )
        return json.loads(raw_response)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI 응답 형식이 올바르지 않습니다.",
        )
    except Exception as e:
        print(f"[Error] API 호출 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))