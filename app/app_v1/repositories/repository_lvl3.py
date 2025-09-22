import logging
import statistics
import time
import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from starlette.responses import StreamingResponse, JSONResponse, HTMLResponse

from app.app_v1.core import OPENROUTER_API_KEY
from app.app_v1.core.config import fetch_available_models
from app.app_v1.schemas import GenerateRequest

router = APIRouter(tags=['Level 3'])

AVAILABLE_MODELS = fetch_available_models()


async def stream_openrouter(request: GenerateRequest, max_tokens: int = 512):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель {request.model} недоступна. Используйте только бесплатные модели: {AVAILABLE_MODELS}")
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

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code == 429:
                    raise HTTPException(status_code=429, detail="Rate limit от OpenRouter")

                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    yield f"data: {line}\n\n"

                yield "event: done\ndata: \n\n"

        except httpx.HTTPStatusError as e:
            logging.exception("HTTP error during streaming from OpenRouter: %s", str(e))
            raise HTTPException(status_code=500, detail="Ошибка при запуске streaming Benchmark.")
        except Exception as e:
            logging.exception("Exception during stream_openrouter")
            yield f"event: error\ndata: {str(e)}\n\n"


@router.post("/generate", summary="Использовать нейросеть V3",
             description="Эндпоинт для использования нейронной сети. "
                         "Необходимо ввести promt, выбрать модель из списка эндпоинта models, отметить bool флаг "
                         "stream который отвечает за SSE-стриминг.")
async def generate_text(request: GenerateRequest, stream: bool = False, max_tokens: int = 512):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель {request.model} недоступна. Используйте только бесплатные модели: {AVAILABLE_MODELS}"
        )

    if stream:
        generator = stream_openrouter(request=request, max_tokens=max_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")

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

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit от OpenRouter")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logging.exception("HTTP error in generate_text")
        raise HTTPException(status_code=500, detail=f"Ошибка от OpenRouter - {e}")
    except Exception as e:
        logging.exception("Error in generate_text")
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к OpenRouter - {e}")

    end = time.perf_counter()

    choices = data.get("choices", [])
    text = (choices[0].get("message", {}).get("content") if choices else None)
    if text is None:
        raise HTTPException(status_code=500, detail="Ответ модели некорректен")

    return JSONResponse({
        "response": text,
        "tokens_used": data.get("usage", {}).get("total_tokens"),
        "latency_seconds": round(end - start, 3)
    })


@router.post("/benchmark", summary="Собрать статистику. promt из .txt",
             description="Эндпоинт для использования нейронной сети. "
                         "Необходимо загрузить .txt файл с записанными promt по строкам, выбрать модель из списка "
                         "эндпоинта models, какое количество раз будет запущен эндпоинт и bool флаг визуализации HTML "
                         "страницы."
                         "Возвращает ответ нейронной сети, статистику latency(avg, min, max, std_dev) и "
                         "DataFrame-HTML страницы.")
async def benchmark_model(prompt_file: UploadFile = File(...), model: str = Form(...), runs: int = Form(5),
                          visualize: bool = Form(False)):
    prompts = (await prompt_file.read()).decode("utf-8").splitlines()
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for prompt in prompts:
            latencies = []
            for _ in range(runs):
                start = time.perf_counter()

                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 128,
                    },
                )
                resp.raise_for_status()
                _ = resp.json()
                end = time.perf_counter()
                latencies.append(end - start)
            stats = {
                "prompt": prompt,
                "avg": round(statistics.mean(latencies), 3),
                "min": round(min(latencies), 3),
                "max": round(max(latencies), 3),
                "std_dev": round(statistics.pstdev(latencies), 3),
            }
            results.append(stats)

    df = pd.DataFrame(results)
    csv_path = "benchmark_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    if visualize:
        table_html = df.to_html(classes="table table-striped table-sm", index=False, escape=False)
        html = f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
              <title>Benchmark results</title>
            </head>
            <body class="p-4">
              <div class="container">
                <h1>Benchmark results for model: {model}</h1>
                <p>runs = {runs}</p>
                {table_html}
                <a href="/download/benchmark_results.csv" class="btn btn-primary">Download CSV</a>
              </div>
            </body>
            </html>
            """
        return HTMLResponse(content=html, status_code=200)
    return results
