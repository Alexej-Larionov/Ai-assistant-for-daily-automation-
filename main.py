
import json
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException, Body
from openai import OpenAI, OpenAIError, APIConnectionError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from uuid import uuid4
#from key import api_key, base_url_flex, models_flex

api_key = None
base_url_flex = "https://litellm.happyhub.ovh"
models_flex = ["deepseek/deepseek-v4-flash", "qwen-3.8-27b", "gpt-oss-120b", "nvidia/nemotron-3-embed-1b"]

app = FastAPI()
client = OpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio"
                )

MODEL = "local"
TITLE = r"^[А-Яа-яЁёA-Za-z-]+( [А-Яа-яЁёA-Za-z-]+){2,10}$"
DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"

#Временное хранилище заметок пока не подключим БД 
notes = {}

Importance = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Category = Literal["работа", "учёба", "личное", "другое"]


#Проверяем доступность локального сервера

try:
    client.with_options(timeout=2, max_retries=0).models.list()
    print("Using local LLM")

except APIConnectionError:
    client = None
    MODEL = models_flex[0] # ["deepseek/deepseek-v4-flash", "qwen-3.8-27b", "gpt-oss-120b", "nvidia/nemotron-3-embed-1b"]
    print("Using remote LLM")


# Классы составляющие заметку

class Metadata(BaseModel):
    deadline: date | None = None
    importance: Importance | None = None
    category: Category | None = None


class Note(BaseModel):
    title: str | None = Field(default=None)
    message: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=Metadata)
    summary: str | None = None


class Generated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(pattern=TITLE)
    summary: str
    deadline: str | None = Field(pattern=DATE)
    importance: Importance
    category: Category


class Generation(BaseModel):
    temperature: float = Field(default=0.3, ge=0, le=10)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=1024, ge=64, le=4096)


class NoteRequest(BaseModel):
    note: Note
    generation: Generation = Field(default_factory=Generation)


class SavedNote(Note):
    id: str


#ввод ключа    

@app.post("/api-key")
def set_api_key(key: str = Body(embed=True)):
    global api_key
    api_key = key
    return {"status": "ok"}
    
    
#Ограничиваем ответы LLM как с помощью промптов, так и с помощью constrained decoding через json schema

def generate(note, generation):
    #Если локального сервера нет, проверяем подключение у даленному по ключу
    
    llm = client

    if llm is None:
        if api_key is None:
            raise HTTPException(503, "API key required")

        llm = OpenAI(base_url=base_url_flex, api_key=api_key)
    
        
    #запрос к LLM
    
    response = client.chat.completions.create(
        model=MODEL,
        temperature=generation.temperature,
        top_p=generation.top_p,
        max_tokens=generation.max_tokens,
        extra_body={"top_k": generation.top_k}
            if MODEL == "local" and generation.top_k is not None else {},
        messages=[
            {
                "role": "system",
                "content": """
                Заполни недостающие поля заметки.
                title: до 11 слов, кратко описывающих тему заметки.
                summary: краткое содержание заметки в 1-2 предложениях.
                importance: от 1 (низкая) до 10 (критическая).
                category: одна из допустимых категорий отражающих суть текста.
                deadline: дата для напоминания о заметке в формате YYYY-MM-DD или null, если срок неизвестен.
                Не изменяй пользовательские значения.
                Не придумывай отсутствующие факты и даты.
                """
            },
            {
                "role": "user",
                "content": json.dumps({
                    "today": date.today().isoformat(),
                    "note": note.model_dump(mode="json")
                }, ensure_ascii=False)
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Generated",
                "strict": True,
                "schema": Generated.model_json_schema()
            }
        }
    )

    return Generated.model_validate_json(response.choices[0].message.content)


#Добавление заметки, отправляет заметку LLM, дополянет пропущеные поля, возвращает заполненую заметку. Поле message не может быт ьпустым и должно быть заполнено пользователем. 

@app.post("/notes", response_model=SavedNote)
def create_note(request: NoteRequest):
    
    note = request.note

    try:
        result = generate(note, request.generation)
        data = note.model_dump()
        
        # Дополняем только незаполненные поля заметки
        for field in ("title", "summary"):
            if data[field] is None:
                data[field] = getattr(result, field)

        for field in ("deadline", "importance", "category"):
            if data["metadata"][field] is None:
                data["metadata"][field] = getattr(result, field)

        note = SavedNote(id=str(uuid4()), **data)
        notes[note.id] = note

        return note
    except (OpenAIError, ValidationError, ValueError) as e:
        raise HTTPException(status_code=502, detail=str(e))


#Вывести заметку по id 

@app.get("/notes/{id}", response_model=SavedNote)
def get_note(id: str):
    if id not in notes:
        raise HTTPException(status_code=404, detail="Note not found")

    return notes[id]


#вывести все сохраненные заметки в сессии

@app.get("/notes", response_model=list[SavedNote])
def get_notes():
    return list(notes.values())