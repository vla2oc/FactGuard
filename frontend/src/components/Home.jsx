import { useMemo, useState } from "react";
import CalendarDemo from "@/dashbord/widget/CalendarDemo";
import { useEvents } from "@/hooks/useEvents";
import { iso } from "@/utils/date";
import MapSingle from "./MapSingle";

export default function Home() {
  // диапазон по умолчанию: 1-й день текущего месяца → сегодня
  const today = new Date();
  const firstOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);

  const [startDate, setStartDate] = useState(firstOfMonth);
  const [endDate, setEndDate] = useState(today);

  // опциональные фильтры
  const [eventType, setEventType] = useState("");
    const [subEvent, setSubEvent] = useState("");

  // загрузка событий с бэка
  const { data: events, loading, error } = useEvents({
    startDate,
    endDate,
    eventType,
    subEvent,
    
  });

  // ISO-дни, где есть события — для подсветки в календаре
  const eventDays = useMemo(() => {
    const s = new Set();
    for (const e of events) {
      const d = e?.ts || e?.date
      if (e?.ts) s.add(String(e.ts).slice(0, 10));
    }
    return s;
  }, [events]);

  // функция для CalendarDemo: true, если в этот день есть события
  const isEventDay = (date) => eventDays.has(iso(date));

  // отсортируем события по времени
  const sorted = useMemo(
    () => [...events].sort((a, b) => String(a.ts).localeCompare(String(b.ts))),
    [events]
  );
  const stop = (e) => e.stopPropagation(); // чтобы карта не зумилась под панелью

  return (
    <div className="fixed inset-0"> 
      <MapSingle events={events}>

        
        {/* Всё, что положишь сюда, окажется панелью поверх карты */}
        <div className="">
        <div className=" bg-stone-900 rounded-lg">
          <div className="text-lg text-gray-300 flex flex-row justify-center items-center gap-2" onWheel={stop} onMouseDown={stop} onTouchStart={stop}>
            Period: <b>{iso(startDate)}</b> — <b>{iso(endDate)}</b>
            {loading ? " (loading...)" : ""}
          </div>
          <CalendarDemo
            startDate={startDate}
            endDate={endDate}
            setStartDate={setStartDate}
            setEndDate={setEndDate}
            isEventDay={isEventDay}
          />
        </div>
        </div>
      </MapSingle>

      {error && (
        <div className="absolute bottom-4 left-4 z-[1200] text-red-400 text-sm bg-black/60 px-2 py-1 rounded">
          Error: {String(error)}
        </div>
      )}
    </div>
  );
}