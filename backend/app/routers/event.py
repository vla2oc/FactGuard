from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Optional, Annotated

from ..schemas.event import EventQueryParams, EventOut
from ..services.EventService import EventService

events_router = APIRouter(prefix="/events", tags=["events"])

# Создаем единственный экземпляр сервиса
event_service = EventService()


@events_router.get("/", response_model=List[EventOut])
async def get_events(params: Annotated[EventQueryParams, Query()]):
    return event_service.get_events(params)


@events_router.get("/{event_id}", response_model=EventOut)
async def get_event_by_id(event_id: int):
    """Получить событие по ID"""
    return event_service.get_event_by_id(event_id)
