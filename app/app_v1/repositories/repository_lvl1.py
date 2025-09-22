import logging
import requests
from fastapi import APIRouter, HTTPException
from typing import List

from app.app_v1.core import OPENROUTER_API_KEY
from app.app_v1.core.config import fetch_available_models
from app.app_v1.schemas import GenerateRequest

router = APIRouter(tags=['Level 1'])

AVAILABLE_MODELS = fetch_available_models()


@router.get("/models", response_model=List[str], summary="Получить список всех моделей",
            description="Эндпоинт для получения списка всех доступных моделей нейронных сетей.")
def get_models():
    return AVAILABLE_MODELS


@router.post("/generate", summary="Использовать нейросеть V1",
             description="Эндпоинт для использования нейронной сети. "
                         "Необходимо ввести promt и выбрать модель из списка эндпоинта models")
def generate_text(request: GenerateRequest):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель {request.model} недоступна. Используйте только бесплатные модели: {AVAILABLE_MODELS}"
        )
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        return {"response": data["choices"][0]["message"]["content"]}

    except Exception as e:
        logging.error(f"Ошибка при генерации текста: %s", str(e))
        raise HTTPException(status_code=500, detail="Ошибка при обращении к OpenRouter API.")
