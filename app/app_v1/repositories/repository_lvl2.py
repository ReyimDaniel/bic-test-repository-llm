import logging
import time
import math
import requests
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.app_v1.core import OPENROUTER_API_KEY
from app.app_v1.core.config import fetch_available_models
from app.app_v1.schemas import GenerateRequest

router = APIRouter(tags=['Level 2'])

AVAILABLE_MODELS = fetch_available_models()


def call_openrouter(request: GenerateRequest, max_tokens: int = 512):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель {request.model} недоступна. Используйте только бесплатные модели: {AVAILABLE_MODELS}")
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": max_tokens,
        }

        retry_delay = 1
        for attempt in range(5):
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                logging.warning(f"429 Error. Too Many Requests. Повтор через {retry_delay}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logging.error(f"Ошибка при генерации текста: %s", str(e))
        raise HTTPException(status_code=500, detail="Ошибка при обращении к OpenRouter API")

    raise HTTPException(status_code=429, detail="Слишком много запросов. Попробуйте позже.")


@router.post("/generate", summary="Использовать нейросеть V2",
             description="Эндпоинт для использования нейронной сети. "
                         "Необходимо ввести promt и выбрать модель из списка эндпоинта models. "
                         "Возвращает ответ нейронной сети, количество использованных токенов из API и затраченное "
                         "время.")
def generate_text(request: GenerateRequest, max_tokens: int = 512):
    try:
        start = time.time()
        data = call_openrouter(request=request, max_tokens=max_tokens)
        end = time.time()

        return {
            "response": data["choices"][0]["message"]["content"],
            "tokens_used": data.get("usage", {}).get("total_tokens", None),
            "latency_seconds": end - start
        }

    except Exception as e:
        logging.error(f"Ошибка при генерации текста: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при обращении к OpenRouter API")


@router.post("/benchmark", summary="Собрать статистику. promt из .txt",
             description="Эндпоинт для использования нейронной сети. "
                         "Необходимо загрузить .txt файл с записанными promt по строкам и выбрать модель из списка "
                         "эндпоинта models и выбрать какое количество раз будет запущен эндпоинт. "
                         "Возвращает ответ нейронной сети, статистику latency(avg, min, max, std_dev) и сохраняет эти "
                         "результаты в таблице .csv")
async def identify_benchmark(model: str, runs: int, prompt_file: UploadFile = File(...)):
    try:
        contents = await prompt_file.read()
        prompts = contents.decode("utf-8").splitlines()

        results = []
        for prompt in prompts:
            latencies = []
            for i in range(runs):
                start = time.time()
                request = GenerateRequest(prompt=prompt, model=model)
                call_openrouter(request=request, max_tokens=512)
                end = time.time()
                latencies.append(end - start)

            stats = {
                "prompt": prompt,
                "avg": sum(latencies) / runs,
                "min": min(latencies),
                "max": max(latencies),
                "std_dev": math.sqrt(sum((x - (sum(latencies) / runs)) ** 2 for x in latencies) / runs)
            }
            results.append(stats)

        df = pd.DataFrame(results)
        df.to_csv("benchmark_results.csv", index=False)
        return {"results": results}

    except Exception as e:
        logging.error(f"Ошибка при вычислении метрик: %s", str(e))
        raise HTTPException(status_code=500, detail="Ошибка при запуске Identify Benchmark.")
