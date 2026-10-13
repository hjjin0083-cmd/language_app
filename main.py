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

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key) if api_key else None


class SimpleLanguageDetail(BaseModel):
    text: str = Field(description="해당 언어로 번역/보정된 문장")
    pronunciation: Optional[str] = Field(
        default=None,
        description="발음 안내 (중국어 병음/성조, 일본어 히라가나 등)",
    )


class UserAnalysisDict(BaseModel):
    korean: str = Field(description="사용자가 입력한 문장의 자연스러운 한국어 다듬기")
    english: SimpleLanguageDetail = Field(description="사용자 입력 문장의 영어 표현")
    chinese: SimpleLanguageDetail = Field(description="사용자 입력 문장의 중국어 표현 및 병음")
    japanese: SimpleLanguageDetail = Field(description="사용자 입력 문장의 일본어 표현 및 발음")


class LanguageDetail(BaseModel):
    text: str = Field(description="해당 언어로 번역/응답된 문장")
    structure: Optional[str] = Field(default=None, description="어순 구조 (SOV, SVO 등)")
    pronunciation: Optional[str] = Field(
        default=None,
        description="발음 안내 (중국어 병음, 일본어 히라가나 포함)",
    )


class LanguagesDict(BaseModel):
    korean: LanguageDetail
    english: LanguageDetail
    chinese: LanguageDetail
    japanese: LanguageDetail


class MetadataDetail(BaseModel):
    contrast_note: Optional[str] = Field(
        default=None,
        description="선택한 언어와 다른 언어 간의 어순, 핵심 문법 및 회화 팁 설명",
    )


class MultiLingualResponse(BaseModel):
    original_input: str
    selected_language: str
    user_analysis: UserAnalysisDict = Field(
        description="사용자가 선택한 언어로 말한 문장에 대한 4개 국어 대조 분석"
    )
    languages: LanguagesDict = Field(
        description="사용자의 말에 대해 4개 국어로 응답하는 AI 대답"
    )
    metadata: MetadataDetail


class ChatMessage(BaseModel):
    role: str  # 'user' 또는 'model'
    content: str


class UserInput(BaseModel):
    text: str
    selected_lang: Optional[str] = "korean"  # 'korean', 'english', 'japanese', 'chinese'
    history: Optional[List[ChatMessage]] = []


@app.get("/")
async def get_ui():
    return FileResponse(BASE_DIR / "index.html")


def call_gemini_api(prompt_text: str, selected_lang: str, history: List[ChatMessage]) -> str:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    recent_history = history[-6:] if history else []
    history_str = ""
    if recent_history:
        history_str = "\n[이전 대화 내역]\n"
        for msg in recent_history:
            role_name = "사용자" if msg.role == "user" else "AI 대답"
            history_str += f"{role_name}: {msg.content}\n"

    lang_names = {
        "korean": "한국어",
        "english": "영어",
        "japanese": "일본어",
        "chinese": "중국어"
    }
    user_lang_name = lang_names.get(selected_lang, "한국어")

    prompt = f"""
당신은 4개 국어(한국어, 영어, 중국어, 일본어)를 자유자재로 구사하는 멀티 AI 튜터입니다.
사용자는 현재 [{user_lang_name}]로 대화를 시도하고 있습니다.

{history_str}
[현재 사용자 입력 ({user_lang_name})]
"{prompt_text}"

[작성 지침]
1. `user_analysis`:
   - 사용자가 [{user_lang_name}]로 말한 문장을 정확히 이해하고, 4개 국어로 자연스럽게 대조 번역 및 보정하세요.
   - 중국어와 일본어는 pronunciation(병음/성조, 히라가나 발음)을 반드시 포함하세요.

2. `languages`:
   - 사용자가 한 말에 대해 친근한 대화 상대로서 4개 국어(한국어, 영어, 중국어, 일본어)로 대답을 작성하세요.
   - 선택된 언어뿐만 아니라 나머지 3개 국어 사용자도 바로 알아들을 수 있도록 각 언어별로 자연스러운 회화체 대답을 구성해야 합니다.
   - 중국어와 일본어는 pronunciation을 작성하세요.

3. `metadata.contrast_note`:
   - 사용자가 선택한 [{user_lang_name}] 표현과 나머지 언어 간의 주요 어순(SOV/SVO) 차이 및 핵심 문법 표현을 한국어로 알기 쉽게 설명하세요.
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
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    try:
        raw_response = await asyncio.wait_for(
            asyncio.to_thread(call_gemini_api, text, payload.selected_lang or "korean", payload.history or []),
            timeout=60.0,
        )
        return json.loads(raw_response)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="응답 시간이 초과되었습니다.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI 응답 형식이 올바르지 않습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
