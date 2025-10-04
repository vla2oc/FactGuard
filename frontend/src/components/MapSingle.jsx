import React, { useMemo, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
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

export default function MapSingle({ events }) {
  // Берём только валидные точки + приводим к числам на всякий случай
  const points = useMemo(() => {
    return (events || [])
      .map(e => ({
        id: e.id,
        lat: Number(e.lat),
        lon: Number(e.lon),
        city: e.city,
        country: e.country,
        type: e.event_type || e.type,
        subType: e.sub_event_type,
        fatalities: e.fatalities,
        eventsCount: e.events, // если есть поле "events" — количество инцидентов в городе
        when: e.ts,            // если есть время
      }))
      .filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon));
  }, [events]);

  return (
    <div className="rounded-lg border overflow-hidden">
      <MapContainer
        center={[49, 32]}   // стартовый центр, дальше FitBounds всё подстроит
        zoom={6}
        style={{ height: "60vh", width: "100%" }}
        scrollWheelZoom
      >
        <TileLayer
          url={import.meta.env.VITE_TILE_URL || "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"}
          attribution="&copy; OpenStreetMap contributors"
        />

        <FitBoundsOnPoints points={points} />

        {points.map((p) => (
          <CircleMarker
            key={p.id ?? `${p.lat},${p.lon}`}
            center={[p.lat, p.lon]}
            radius={6}
          >
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
                  <div style={{ fontSize: 12, marginTop: 4 }}>Events in city: {p.eventsCount}</div>
                )}
                {p.fatalities != null && (
                  <div style={{ fontSize: 12 }}>Fatalities: {p.fatalities}</div>
                )}
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}
