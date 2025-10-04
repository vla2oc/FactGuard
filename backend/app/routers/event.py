from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Optional

from ..schemas.event import EventQueryParams, EventOut
from ..services.EventService import EventService

events_router = APIRouter(prefix="/events", tags=["events"])

# Создаем единственный экземпляр сервиса
event_service = EventService()


def get_event_service() -> EventService:
    """Dependency injection для EventService"""
    return event_service


@events_router.get("/", response_model=List[EventOut])
async def get_events(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    event_subtype: Optional[str] = Query(None),
    regions: Optional[List[str]] = Query(None),
    cities: Optional[List[str]] = Query(None),
    min_fatalities: Optional[int] = Query(None),
    max_fatalities: Optional[int] = Query(None),
    service: EventService = Depends(get_event_service),
):
    """Получить отфильтрованные события"""
    try:
        # Конвертируем строки дат в date объекты
        start_date_obj = None
        end_date_obj = None

        if start_date:
            from datetime import datetime

            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()

        if end_date:
            from datetime import datetime

            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

        params = EventQueryParams(
            start_date=start_date_obj,
            end_date=end_date_obj,
            event_type=event_type,
            event_subtype=event_subtype,
            regions=regions,
            cities=cities,
            min_fatalities=min_fatalities,
            max_fatalities=max_fatalities,
        )

        return service.get_events(params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат даты: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


@events_router.get("/{event_id}", response_model=EventOut)
async def get_event_by_id(
    event_id: int, service: EventService = Depends(get_event_service)
):
    """Получить событие по ID"""
    event = service.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Событие не найдено")
    return event


@events_router.get("/meta/event-types", response_model=List[str])
async def get_event_types(service: EventService = Depends(get_event_service)):
    """Получить все уникальные типы событий"""
    return service.get_unique_event_types()


@events_router.get("/meta/event-subtypes", response_model=List[str])
async def get_event_subtypes(service: EventService = Depends(get_event_service)):
    """Получить все уникальные подтипы событий"""
    return service.get_unique_event_subtypes()


@events_router.get("/meta/regions", response_model=List[str])
async def get_regions(service: EventService = Depends(get_event_service)):
    """Получить все уникальные регионы"""
    return service.get_unique_regions()


@events_router.get("/meta/cities", response_model=List[str])
async def get_cities(service: EventService = Depends(get_event_service)):
    """Получить все уникальные города"""
    return service.get_unique_cities()


@events_router.get("/meta/statistics", response_model=dict)
async def get_statistics(service: EventService = Depends(get_event_service)):
    """Получить статистику по всем данным"""
    return service.get_statistics()
