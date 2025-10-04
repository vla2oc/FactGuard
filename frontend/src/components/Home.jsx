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

  return (
    <div className="p-6 space-y-6">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold text-white">События</h1>

        <select
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
          className="rounded-md border bg-gray-900 text-white px-2 py-1"
          title="event_type"
        >
          <option value="">All Types</option>
          <option value="Battles">Battles</option>
          <option value="Explosions/Remote violence">Explosions/Remote violence</option>
          <option value="Protests">Protests</option>
          <option value="Violence against civilians">Violence against civilians</option>
          <option value="Riots">Riots</option>
          <option value="Strategic developments">Strategic developments</option>
        </select>


        <div className="text-lg text-gray-700 ml-auto">
          Period: <b>{startDate ? iso(startDate) : "—"}</b> —{" "}
          <b>{endDate ? iso(endDate) : "—"}</b> • Events:{" "}
          <b>{events.length}</b> {loading ? " (loading…)" : ""}
        </div>
      </header>

      {/* Календарь диапазона — отдельный компонент */}
      <CalendarDemo
        startDate={startDate}
        endDate={endDate}
        setStartDate={setStartDate}
        setEndDate={setEndDate}
        isEventDay={isEventDay}
      />
      <MapSingle
        events={events}
      />

      {error && (
        <div className="text-red-400 text-sm">
          Error loading: {String(error)}
        </div>
      )}
    </div>
  );
}
