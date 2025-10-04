from pydantic import BaseModel
from datetime import date
from typing import Optional, List


class EventQueryParams(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    event_type: Optional[str] = None
    event_subtype: Optional[str] = None
    # regions: Optional[List[str]] = None
    # cities: Optional[List[str]] = None
    # min_fatalities: Optional[int] = None
    # max_fatalities: Optional[int] = None


class EventOut(BaseModel):
    id: int
    region: str
    country: str
    city: str
    event_type: str
    sub_event_type: str
    events: int
    fatalities: int
    population_exposure: int
    disorder_type: Optional[str] = None
    lat: float
    lon: float


class EventStatistics(BaseModel):
    total_events: int
    total_fatalities: int
    date_range: dict
    unique_regions: int
    unique_cities: int
    unique_event_types: int
