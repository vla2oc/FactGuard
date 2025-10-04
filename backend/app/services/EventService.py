import pandas as pd
import os
from typing import List, Optional
from ..schemas.event import EventOut, EventQueryParams


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

        # Обрабатываем числовые колонки
        numeric_columns = ["events", "fatalities", "population_exposure", "lat", "lon"]
        for col in numeric_columns:
            if col in self.df.columns:
                # Заменяем пустые значения на 0, затем конвертируем в числа
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        # Обрабатываем строковые колонки
        string_columns = [
            "region",
            "country",
            "city",
            "event_type",
            "sub_event_type",
            "disorder_type",
        ]
        for col in string_columns:
            if col in self.df.columns:
                # Заменяем пустые значения на пустую строку
                self.df[col] = self.df[col].fillna("")

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

        # Конвертируем в список EventOut
        events = []
        for idx, row in filtered_df.iterrows():
            try:
                # Безопасное извлечение числовых значений
                events_count = row.get("events", 0)
                fatalities_count = row.get("fatalities", 0)
                population_exp = row.get("population_exposure", 0)
                latitude = row.get("lat", 0.0)
                longitude = row.get("lon", 0.0)

                # Конвертируем в числа, если это еще не сделано
                events_count = int(events_count) if pd.notna(events_count) else 0
                fatalities_count = (
                    int(fatalities_count) if pd.notna(fatalities_count) else 0
                )
                population_exp = int(population_exp) if pd.notna(population_exp) else 0
                latitude = float(latitude) if pd.notna(latitude) else 0.0
                longitude = float(longitude) if pd.notna(longitude) else 0.0

                # Безопасное извлечение строковых значений
                disorder_type_val = row.get("disorder_type", "")
                disorder_type_val = (
                    str(disorder_type_val)
                    if pd.notna(disorder_type_val) and disorder_type_val != ""
                    else None
                )

                event = EventOut(
                    id=int(idx),
                    region=str(row.get("region", "")),
                    country=str(row.get("country", "")),
                    city=str(row.get("city", "")),
                    event_type=str(row.get("event_type", "")),
                    sub_event_type=str(row.get("sub_event_type", "")),
                    events=events_count,
                    fatalities=fatalities_count,
                    population_exposure=population_exp,
                    disorder_type=disorder_type_val,
                    lat=latitude,
                    lon=longitude,
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
            # Безопасное извлечение числовых значений
            events_count = row.get("events", 0)
            fatalities_count = row.get("fatalities", 0)
            population_exp = row.get("population_exposure", 0)
            latitude = row.get("lat", 0.0)
            longitude = row.get("lon", 0.0)

            # Конвертируем в числа, если это еще не сделано
            events_count = int(events_count) if pd.notna(events_count) else 0
            fatalities_count = (
                int(fatalities_count) if pd.notna(fatalities_count) else 0
            )
            population_exp = int(population_exp) if pd.notna(population_exp) else 0
            latitude = float(latitude) if pd.notna(latitude) else 0.0
            longitude = float(longitude) if pd.notna(longitude) else 0.0

            # Безопасное извлечение строковых значений
            disorder_type_val = row.get("disorder_type", "")
            disorder_type_val = (
                str(disorder_type_val)
                if pd.notna(disorder_type_val) and disorder_type_val != ""
                else None
            )

            return EventOut(
                id=event_id,
                region=str(row.get("region", "")),
                country=str(row.get("country", "")),
                city=str(row.get("city", "")),
                event_type=str(row.get("event_type", "")),
                sub_event_type=str(row.get("sub_event_type", "")),
                events=events_count,
                fatalities=fatalities_count,
                population_exposure=population_exp,
                disorder_type=disorder_type_val,
                lat=latitude,
                lon=longitude,
            )
        except (ValueError, TypeError) as e:
            print(f"Ошибка при обработке события {event_id}: {e}")
            return None
