from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.event import events_router

import uvicorn


app = FastAPI(
    title="Ukraine Events API",
    description="API для отображения событий на территории Украины",
    version="1.0.0",
    prefix="/api/",
)

# Настройка CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(events_router)


@app.get("/")
async def root():
    return {"message": "Ukraine Events API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
