import logging
from fastapi import FastAPI
import uvicorn

from app_v1.repositories.repository_lvl1 import router as router_lvl_1
from app_v1.repositories.repository_lvl2 import router as router_lvl_2
from app_v1.repositories.repository_lvl3 import router as router_lvl_3

logging.basicConfig(
    filename="logs/server_logs.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="FastAPI Application V1")
app.include_router(router=router_lvl_1, prefix="/lvl_1")
app.include_router(router=router_lvl_2, prefix="/lvl_2")
app.include_router(router=router_lvl_3, prefix="/lvl_3")

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
