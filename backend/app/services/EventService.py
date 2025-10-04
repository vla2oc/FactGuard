import pandas as pd
import os
from typing import List, Optional
from ..schemas.event import EventQueryParams, EventOut


class EventService:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.data_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "dataset.xlsx"
        )
        self._load_data()

    def _load_data(self):
        """Загружает данные из Excel файла"""
        try:
            self.df = pd.read_excel(self.data_path)
            self._prepare_data()
            print(f"Данные загружены успешно. Количество записей: {len(self.df)}")
        except FileNotFoundError:
            print(f"Файл не найден: {self.data_path}")
            raise
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise

    def _prepare_data(self):
        """Подготавливает данные после загрузки"""
        if self.df is None:
            return

        # Предполагаемые названия колонок на основе ваших данных
        expected_columns = [
            "date",
            "region",
            "country",
            "city",
            "event_type",
            "sub_event_type",
            "events",
            "fatalities",
            "population_exposure",
            "disorder_type",
            "lat",
            "lon",
        ]

        # Если колонки не названы, присваиваем названия
        if len(self.df.columns) == len(expected_columns):
            self.df.columns = expected_columns

        # Конвертируем дату
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        # Конвертируем числовые колонки
        numeric_columns = ["events", "fatalities", "population_exposure", "lat", "lon"]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Добавляем индекс как ID
        self.df.reset_index(drop=True, inplace=True)

    def get_events(self, params: EventQueryParams) -> List[EventOut]:
        """Возвращает отфильтрованные события на основе параметров запроса"""
        if self.df is None:
            return []

        filtered_df = self.df.copy()

        # Фильтр по дате начала
        if params.start_date:
            filtered_df = filtered_df[
                filtered_df["date"] >= pd.to_datetime(params.start_date)
            ]

        # Фильтр по дате окончания
        if params.end_date:
            filtered_df = filtered_df[
                filtered_df["date"] <= pd.to_datetime(params.end_date)
            ]

        # Фильтр по типу события
        if params.event_type:
            filtered_df = filtered_df[
                filtered_df["event_type"].str.contains(
                    params.event_type, case=False, na=False
                )
            ]

        # Фильтр по подтипу события
        if params.event_subtype:
            filtered_df = filtered_df[
                filtered_df["sub_event_type"].str.contains(
                    params.event_subtype, case=False, na=False
                )
            ]

        # Фильтр по регионам
        if params.regions:
            filtered_df = filtered_df[filtered_df["region"].isin(params.regions)]

        # Фильтр по городам
        if params.cities:
            filtered_df = filtered_df[filtered_df["city"].isin(params.cities)]

        # Фильтр по минимальному количеству жертв
        if params.min_fatalities is not None:
            filtered_df = filtered_df[
                filtered_df["fatalities"] >= params.min_fatalities
            ]

        # Фильтр по максимальному количеству жертв
        if params.max_fatalities is not None:
            filtered_df = filtered_df[
                filtered_df["fatalities"] <= params.max_fatalities
            ]

        # Конвертируем в список EventOut
        events = []
        for idx, row in filtered_df.iterrows():
            try:
                event = EventOut(
                    id=int(idx),
                    region=str(row.get("region", "")),
                    country=str(row.get("country", "")),
                    city=str(row.get("city", "")),
                    event_type=str(row.get("event_type", "")),
                    sub_event_type=str(row.get("sub_event_type", "")),
                    events=int(row.get("events", 0)),
                    fatalities=int(row.get("fatalities", 0)),
                    population_exposure=int(row.get("population_exposure", 0)),
                    disorder_type=str(row.get("disorder_type", ""))
                    if pd.notna(row.get("disorder_type"))
                    else None,
                    lat=float(row.get("lat", 0.0)),
                    lon=float(row.get("lon", 0.0)),
                )
                events.append(event)
            except (ValueError, TypeError) as e:
                print(f"Ошибка при обработке строки {idx}: {e}")
                continue

        return events

    def get_event_by_id(self, event_id: int) -> Optional[EventOut]:
        """Возвращает событие по ID"""
        if self.df is None or event_id >= len(self.df) or event_id < 0:
            return None

        row = self.df.iloc[event_id]
        try:
            return EventOut(
                id=event_id,
                region=str(row.get("region", "")),
                country=str(row.get("country", "")),
                city=str(row.get("city", "")),
                event_type=str(row.get("event_type", "")),
                sub_event_type=str(row.get("sub_event_type", "")),
                events=int(row.get("events", 0)),
                fatalities=int(row.get("fatalities", 0)),
                population_exposure=int(row.get("population_exposure", 0)),
                disorder_type=str(row.get("disorder_type", ""))
                if pd.notna(row.get("disorder_type"))
                else None,
                lat=float(row.get("lat", 0.0)),
                lon=float(row.get("lon", 0.0)),
            )
        except (ValueError, TypeError) as e:
            print(f"Ошибка при обработке события {event_id}: {e}")
            return None

    def get_unique_event_types(self) -> List[str]:
        """Возвращает список уникальных типов событий"""
        if self.df is None or "event_type" not in self.df.columns:
            return []
        return self.df["event_type"].dropna().unique().tolist()

    def get_unique_event_subtypes(self) -> List[str]:
        """Возвращает список уникальных подтипов событий"""
        if self.df is None or "sub_event_type" not in self.df.columns:
            return []
        return self.df["sub_event_type"].dropna().unique().tolist()

    def get_unique_regions(self) -> List[str]:
        """Возвращает список уникальных регионов"""
        if self.df is None or "region" not in self.df.columns:
            return []
        return self.df["region"].dropna().unique().tolist()

    def get_unique_cities(self) -> List[str]:
        """Возвращает список уникальных городов"""
        if self.df is None or "city" not in self.df.columns:
            return []
        return self.df["city"].dropna().unique().tolist()

    def get_statistics(self) -> dict:
        """Возвращает базовую статистику по данным"""
        if self.df is None:
            return {}

        return {
            "total_events": len(self.df),
            "total_fatalities": int(self.df["fatalities"].sum())
            if "fatalities" in self.df.columns
            else 0,
            "date_range": {
                "start": self.df["date"].min().strftime("%Y-%m-%d")
                if "date" in self.df.columns and pd.notna(self.df["date"].min())
                else None,
                "end": self.df["date"].max().strftime("%Y-%m-%d")
                if "date" in self.df.columns and pd.notna(self.df["date"].max())
                else None,
            },
            "unique_regions": len(self.get_unique_regions()),
            "unique_cities": len(self.get_unique_cities()),
            "unique_event_types": len(self.get_unique_event_types()),
        }
