import React, { useMemo, useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, Tooltip} from "react-leaflet";
import L from "leaflet";
import BordersLayer from "@/features/BordersLayers";
import "leaflet/dist/leaflet.css";

function FitBoundsOnPoints({ points }) {
  const map = useMap();
  useEffect(() => {
    if (!points?.length) return;
    const bounds = L.latLngBounds(points.map(p => [p.lat, p.lon]));
    // если все точки в одной локации — немного увеличим зум
    if (bounds.isValid()) {
      if (bounds.getNorthEast().equals(bounds.getSouthWest())) {
        map.flyTo(bounds.getNorthEast(), 10, { duration: 0.6 });
      } else {
        map.fitBounds(bounds.pad(0.2), { animate: true });
      }
    }
  }, [points, map]);
  return null;
}
// MapSingle.jsx (добавь рядом с FitBoundsOnPoints)
function EnsureMarkerPane() {
  const map = useMap();
  useEffect(() => {
    if (!map.getPane("markers-top")) {
      map.createPane("markers-top");
      const p = map.getPane("markers-top");
      p.style.zIndex = 680;           // выше областей/контуров и overlayPane
      p.style.pointerEvents = "auto"; // кликабельно
    }
  }, [map]);
  return null;
}

const truncate = (s, n = 120) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s);


export default function MapSingle({ events, children }) {

const STADIA_DARK = "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png";
  // 1) нормализуем входные данные
 const points = useMemo(() => {
  const byLoc = new Map();
  for (const e of (events || [])) {
    const lat = Number(e.lat), lon = Number(e.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const key = `${lat.toFixed(4)},${lon.toFixed(4)}`; // грид ~11м (на твоё усмотрение)
    const cur = byLoc.get(key) || {
      id: key, lat, lon, city: e.city, country: e.country,
      type: e.event_type || e.type, subType: e.sub_event_type,
      eventsCount: 0, fatalities: 0, items: [],
      summary: "",
      when: null,
    };

    const ec = Number(e.events) || 1;
    const ft = Number(e.fatalities) || 0;
    cur.eventsCount += ec;
    cur.fatalities += ft;
    if (!cur.when && e.ts) cur.when = e.ts;
    if (!cur.summary && (e.summary || e.description)) cur.summary = e.summary || e.description;
    cur.items.push(e);

    byLoc.set(key, cur);
  }
  return [...byLoc.values()];
}, [events]);

  // 2) найдём максимум по количеству событий — для нормализации 0..1
  const maxEvents = useMemo(() => {
    const arr = points.map(p => Number(p.eventsCount ?? 0));
    return Math.max(1, ...arr);
  }, [points]);

  // 3) функции цвета/радиуса по t в [0..1]
  const colorByT = (t) => {
    // тёплая палитра: жёлтый (мало) → красный (много)
    const hue = 50 - 50 * t; // 50° → 0°
    return `hsl(${hue}, 95%, 50%)`;
  };
  const radiusByT = (t) => 4 + Math.round(12 * t); // 4..16

  return (
    <div className="fixed inset-0">
      <MapContainer
        center={[49, 32]}
        zoom={6}
        className="w-full h-full"
        scrollWheelZoom
      >
       <TileLayer
  url={STADIA_DARK}
  attribution='&copy; Stadia Maps, &copy; OpenMapTiles, &copy; OpenStreetMap contributors'
/>

        <BordersLayer showOblasts />
        <EnsureMarkerPane />
        <FitBoundsOnPoints points={points} />

        {points.map((p) => {
          const raw = p.eventsCount ?? 0;
          const val = Number(raw);
          const c = Number.isFinite(val) ? val : 0;

          const t = Math.max(0, Math.min(1, c / maxEvents)); // нормализация
          const fill = colorByT(t);
          const r = radiusByT(t);

          return (
            <CircleMarker
              key={p.id ?? `${p.lat},${p.lon}`}
              center={[p.lat, p.lon]}
              pane="markers-top"
              radius={r}
              pathOptions={{ color: fill, fillColor: fill, fillOpacity: 0.85, weight: 1 }}
            >
                <Tooltip direction="top" offset={[0, -8]} opacity={1} sticky>
    <div style={{ fontSize: 14, lineHeight: 1.2 }}>
      <b>{p.type || "Event"}</b>{p.subType ? ` · ${p.subType}` : ""}
      <div>{[p.city, p.country].filter(Boolean).join(", ")}</div>
      {Number.isFinite(p.eventsCount) && <div>Events: {p.eventsCount}</div>}
      {Number.isFinite(p.fatalities) && <div><b className="text-red-500">Fatalities:</b> <span className="text-red-900">{p.fatalities}</span></div>}
      {p.summary && <div style={{ marginTop: 4 }}>{truncate(p.summary, 120)}</div>}
    </div>
  </Tooltip>

              
              <Popup>
                <div style={{ maxWidth: 260 }}>
                  <div><b>{p.type || "Event"}</b>{p.subType ? ` · ${p.subType}` : ""}</div>
                  <div style={{ fontSize: 12, opacity: .8 }}>
                    {[p.city, p.country].filter(Boolean).join(", ")}
                  </div>
                  {p.when && (
                    <div style={{ fontSize: 12, opacity: .7 }}>
                      {new Date(p.when).toLocaleString("en-GB")}
                    </div>
                  )}
                  {p.eventsCount != null && (
                    <div style={{ fontSize: 12, marginTop: 4 }}>
                      Events in city: {p.eventsCount}
                    </div>
                  )}
                  {p.fatalities != null && (
                    <div style={{ fontSize: 12 }}><span className="text-red-500">Fatalities:</span> <span className="text-red-900">{p.fatalities}</span></div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
      {children && (
        <div
          className="absolute top-2 left-2 z-[1200] max-w-full"
          onWheel={stop}
          onMouseDown={stop}
          onTouchStart={stop}
        >
          <div className="bg-slate-800 rounded-xl text-white  shadow-xl">
            {children}
          </div>
        </div>
      )}
    </div>
  );
}