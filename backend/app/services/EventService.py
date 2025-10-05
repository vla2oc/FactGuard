import pandas as pd
import numpy as np
import os
from typing import List, Optional

from ..schemas.event import EventQueryParams, EventOut
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import PoissonRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime


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

    def statistic(self,name, columns):
        if name == 'avg':
            try:
                return np.mean(self.df[columns],axis=0)
            except:
                return "Provided column isn't numerical"  
        elif name == 'std':
            try:
                return np.std(self.df[columns],axis=0)
            except:
                return "Provided column isn't numerical"
        elif name == 'sum':
            try:
                return np.sum(self.df[columns],axis=0)
            except:
                return "Provided column isn't numerical"
        else:
            return 'Wrong stat'
    
    def statistic_over(self, columns, statistics, oblast=None, start_date=None, end_date=None, event_types=None):
        if start_date:
            df_filterd = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
            return self.statistic(statistics,df_filterd,columns)
        elif oblast:
            df_region = self.df[self.df["city"] == oblast]
            return self.statistic(statistics,df_region, columns)
        elif event_types:
            if type(event_types) != list():
                event_types = [event_types]
            stat = dict()
            for event in event_types:
                df_temp = self.df[self.df['event_type'] == event][columns]
                stat[event] = self.statistic(statistics, df_temp, columns)
            return stat
    
    def predict_event_count(self,df=None,city=None):
        df_long = self.df[['WEEK','ADMIN1']].copy()
        df_long.head()
        if city:
            cities = [city]
        else:
            cities = df_long['ADMIN1'].unique()
        event_count = dict()
        for city in cities:
            model = ARIMA(df_long[df_long['ADMIN1']==city]['WEEK'].value_counts().sort_index(), order=(2, 1, 2))  # you can tune (p,d,q)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            event_count[city] = np.round(forecast.iloc[0])
        return event_count

    def predict_coord(self,df=None,city=None):

        df_coord = self.df[['WEEK', 'ADMIN1', 'CENTROID_LATITUDE', 'CENTROID_LONGITUDE']].copy()

        encoder = OneHotEncoder(sparse_output=False)
        X_admin = encoder.fit_transform(self.df[['ADMIN1']])

        df_coord['week_num'] = df_coord['WEEK'].dt.isocalendar().week

        X = np.hstack([X_admin, df_coord[['week_num']].values])
        y = self.df[['CENTROID_LATITUDE', 'CENTROID_LONGITUDE']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train) 

        y_pred = model.predict(X_test)

        next_week = pd.to_datetime(df_coord['WEEK'].iloc[-1] + datetime.timedelta(days=7)) 

        week_num = next_week.isocalendar()[1]
        cities = df_coord['ADMIN1'].unique()

        coord = dict()

        if city:
            cities = [city]
        else:
            cities = df_coord['ADMIN1'].unique()

        for city in cities:
            X_new = np.hstack([encoder.transform([[city]]), [[week_num]]])
            pred_coords = model.predict(X_new)
            coord[city] = pred_coords[0]

        return coord
    
    def create_prediction_df(self, df=None):
        df_coord = self.predict_coord()
        df_event_count = self.predict_event_count()
        rows = []
        for city, count in df_event_count.items():
            for i in range(int(count)):
                rows.append([city])

        df_pred = pd.DataFrame(rows, columns=['City'])
        df_pred['WEEK_NUM'] = 39

        df_pred['CENTROID_LATITUDE'] = df_pred['City'].apply(lambda x: df_coord[x][0])
        df_pred['CENTROID_LONGITUDE'] = df_pred['City'].apply(lambda x: df_coord[x][1])

        lags = 3
        lag_values = {}

        for city, group in self.df.groupby('ADMIN1'):

            group = group.sort_values('WEEK')


            last_vals = group['EVENTS'].tail(lags).tolist()

            lag_values[city] = last_vals

        for col in ['EVENTS', 'FATALITIES']:
            for i in range(1, lags+1):
                df_pred[f'{col}_lag{i}'] = df_pred['City'].apply(lambda city: lag_values[city][-i])


        for col in ['EVENTS', 'FATALITIES']:
            df_pred[f'{col}_rolling_mean'] = (
                self.df.sort_values(['ADMIN1', 'WEEK'])      
                  .groupby('ADMIN1')[col]                
                  .transform(lambda x: x.shift(1)        
                                     .rolling(3, min_periods=1)
                                     .mean())
            )

        rolling_cols = ['EVENTS_rolling_mean', 'FATALITIES_rolling_mean']
        df_pred[rolling_cols] = df_pred.groupby('City')[rolling_cols].transform(lambda x: x.bfill().ffill())

    
        for col in rolling_cols:
            df_pred.iloc[-1, df_pred.columns.get_loc(col)] = df_pred.iloc[-2][col]

        df_pred = pd.get_dummies(df_pred, columns=['City'], prefix='ADMIN1')

        df_pred = df_pred[sorted(df_pred.columns)]

        return df_pred
    
    def predict_event_type(self, df=None,unencoded=False):
        df_event = self.df[['WEEK', 'ADMIN1', 'EVENT_TYPE', 'EVENTS', 'FATALITIES', 'CENTROID_LATITUDE', 'CENTROID_LONGITUDE']].copy()
    
        lags = 3 
        for col in ['EVENTS','FATALITIES']:
            for lag in range(1, lags+1):
                df_event[f'{col}_lag{lag}'] = df_event.groupby('ADMIN1')[col].shift(lag)


        lag_cols = [f'{col}_lag{lag}' for col in ['EVENTS','FATALITIES'] for lag in range(1, lags+1)]

        window = 3 


        for col in ['EVENTS', 'FATALITIES']:
            df_event[f'{col}_rolling_mean'] = df_event.groupby('ADMIN1')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        df_event[lag_cols] = df_event.groupby('ADMIN1')[lag_cols].transform(lambda x: x.bfill())

        cols_to_fill = [col for col in df_event.columns if 'lag' in col or 'rolling' in col]


        df_event[cols_to_fill] = df_event.groupby('ADMIN1')[cols_to_fill].transform(lambda x: x.bfill().ffill())


        df_event = df_event.drop(['EVENTS','FATALITIES'],axis=1)


        df_event = pd.get_dummies(df_event, columns=['ADMIN1'], prefix='ADMIN1')
        df_event = pd.get_dummies(df_event, columns=['EVENT_TYPE'], prefix='EVENT_TYPE')
        df_event['WEEK_NUM'] = df_event['WEEK'].dt.isocalendar().week
        df_event = df_event.drop('WEEK',axis=1)

        admin_cols = [col for col in df_event.columns if col.startswith('ADMIN1_')]
        event_type_cols = [col for col in df_event.columns if col.startswith('EVENT_TYPE_')]
        feature_cols = ['WEEK_NUM', 'CENTROID_LATITUDE', 'CENTROID_LONGITUDE','EVENTS_rolling_mean', 'FATALITIES_rolling_mean'] + [f'{col}_lag{lag}' for col in ['EVENTS','FATALITIES'] for lag in range(1, lags+1)] + admin_cols



        X = df_event[sorted(feature_cols)]
        y = df_event[event_type_cols]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))

        model.fit(X_train, y_train)


        X_new = self.create_prediction_df()

        y_pred = model.predict(X_new)
        event_types = ['Battles', 'Explosions/Remote violence', 'Protests', 'Riots', 'Strategic developments', 'Violence against civilians']

        if unencoded:
            y_pred_combined = []
            for i in range(len(y_pred)):
                for j, col in enumerate(event_types):
                    if y_pred[i][j]:   
                        y_pred_combined.append(event_types[j])
                        break
            return y_pred_combined
        else:
            return pd.DataFrame(y_pred, columns=event_types)
        
    def prediction_df(self,df=None):
        x = self.create_prediction_df()
        y = self.predict_event_type()

        return pd.concat([x,y], axis=1)
    
    def predict_fatalities(self, X_new, df=None):
        df_fatal = self.df[['ADMIN1','WEEK', 'CENTROID_LATITUDE', 'CENTROID_LONGITUDE', 'EVENTS', 'FATALITIES','EVENT_TYPE']].copy()
        lags = 3 
        for col in ['EVENTS','FATALITIES']:
            for lag in range(1, lags+1):
                df_fatal[f'{col}_lag{lag}'] = df_fatal.groupby('ADMIN1')[col].shift(lag)
        lag_cols = [f'{col}_lag{lag}' for col in ['EVENTS','FATALITIES'] for lag in range(1, lags+1)]
        window = 3 
        for col in ['EVENTS', 'FATALITIES']:
            df_fatal[f'{col}_rolling_mean'] = df_fatal.groupby('ADMIN1')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        df_fatal[lag_cols] = df_fatal.groupby('ADMIN1')[lag_cols].transform(lambda x: x.bfill())
        cols_to_fill = [col for col in df_fatal.columns if 'lag' in col or 'rolling' in col]
        df_fatal[cols_to_fill] = df_fatal.groupby('ADMIN1')[cols_to_fill].transform(lambda x: x.bfill().ffill())

        df_fatal = pd.get_dummies(df_fatal, columns=['ADMIN1'], prefix='ADMIN1')
        df_fatal = pd.get_dummies(df_fatal, columns=['EVENT_TYPE'], prefix='EVENT_TYPE')

        df_fatal['WEEK_NUM'] = df_fatal['WEEK'].dt.isocalendar().week
        df_fatal = df_fatal.drop('WEEK',axis=1)

        admin_cols = [col for col in df_fatal.columns if col.startswith('ADMIN1_')]
        event_cols = [col for col in df_fatal.columns if col.startswith('EVENT_TYPE_')]
        feature_cols = ['WEEK_NUM', 'CENTROID_LATITUDE', 'CENTROID_LONGITUDE','EVENTS_rolling_mean', 'FATALITIES_rolling_mean'] + [f'{col}_lag{lag}' for col in ['EVENTS','FATALITIES'] for lag in range(1, lags+1)] + admin_cols + event_cols

        X = df_fatal[sorted(feature_cols)]
        y_events = df_fatal['EVENTS']
        y_fatalities = df_fatal['FATALITIES']

        X_train, X_test, y_train_events, y_test_events = train_test_split(X, y_events, test_size=0.2, shuffle=False)
        _, _, y_train_fatal, y_test_fatal = train_test_split(X, y_fatalities, test_size=0.2, shuffle=False)

        params = {
            'objective': 'poisson',   
            'metric': 'rmse',         
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'seed': 42
        }

        events_model = LGBMRegressor(
            objective='poisson',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            random_state=42
        )

        events_model.fit(
            X_train, y_train_events,
            eval_set=[(X_test, y_test_events)],
            eval_metric='rmse',

        )

        # FATALITIES model
        fatal_model = LGBMRegressor(
            objective='poisson',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            random_state=42
        )

        fatal_model.fit(
            X_train, y_train_fatal,
            eval_set=[(X_test, y_test_fatal)],
            eval_metric='rmse',

        )

        y_pred_events = events_model.predict(X_new)
        y_pred_fatal = fatal_model.predict(X_new)

        # Clip predictions to avoid negative counts
        y_pred_events = pd.DataFrame(np.round(np.clip(y_pred_events, 0, None)),columns=['Events'])
        y_pred_fatal = pd.DataFrame(np.round(np.clip(y_pred_fatal, 0, None)), columns=['Fatalities'])

        prediction = pd.concat([y_pred_events, y_pred_fatal],axis=1)
        return prediction
    
    def predict(self, df=None):
        x = self.prediction_df()

        y = self.predict_fatalities(x)

        pred = pd.concat([x, y], axis=1)

        event_cols = ['Battles', 'Explosions/Remote violence', 'Protests', 
                  'Riots', 'Strategic developments', 'Violence against civilians']

        pred['EVENT_TYPE'] = pred[event_cols].idxmax(axis=1)

        admin_cols = [col for col in pred.columns if col.startswith('ADMIN1_')]

        pred['ADMIN1'] = pred[admin_cols].idxmax(axis=1).str.replace('ADMIN1_', '')

        cols_to_drop = event_cols + admin_cols +['EVENTS_lag1', 'EVENTS_lag2', 'EVENTS_lag3', 'EVENTS_rolling_mean', 'FATALITIES_lag1', 'FATALITIES_lag2', 'FATALITIES_lag3', 'FATALITIES_rolling_mean']

        year = 2025
        week = 39
        week_end = datetime.date.fromisocalendar(year, week, 7)
        pred = pred.drop(columns=cols_to_drop)
        pred['WEEK_NUM'] = week_end

        new_order=['WEEK_NUM', 'ADMIN1','EVENT_TYPE','Events','Fatalities','CENTROID_LATITUDE','CENTROID_LONGITUDE']

        pred = pred[new_order]

        return pred
    
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
