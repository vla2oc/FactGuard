import React, { useEffect, useState } from "react";
import { GeoJSON, useMap } from "react-leaflet";

export default function BordersLayer() {
  const map = useMap();
  const [ua, setUa] = useState(null);

  useEffect(() => {
    fetch("/geo/ua-country.geo.json").then(r => r.json()).then(setUa);
  }, []);

  // отдельные «панели», чтобы контур был над тайлами/маркерами
  useEffect(() => {
    if (!map) return;
    if (!map.getPane("ua-glow")) {
      map.createPane("ua-glow");
      map.getPane("ua-glow").style.zIndex = 340; // под контуром
    }
    if (!map.getPane("ua-edge")) {
      map.createPane("ua-edge");
      map.getPane("ua-edge").style.zIndex = 350; // сам контур
    }
    if (!map.getPane("ua-oblasts")) {
      map.createPane("ua-oblasts");
      map.getPane("ua-oblasts").style.zIndex = 355; // между свечением и контуром
    }
  }, [map]);

  if (!ua) return null;

  return (
    <>
      {/* «Свечение» под границей */}
      <GeoJSON
        data={ua}
        pane="ua-glow"
        style={{ color: "#22d3ee", weight: 4, opacity: 0.35, fillOpacity: 0 }}
      />
      {/* Тонкий яркий контур */}
      <GeoJSON
        data={ua}
        pane="ua-edge"
        style={{
          color: "#A5F3FC", weight: 1.5, opacity: 0.95, fillOpacity: 0,
          dashArray: null, lineJoin: "round"
        }}
      />
    </>
  );
}
