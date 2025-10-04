import { useEffect, useState } from "react";
import { getEvents } from "@/utils/eventsApi";
import { toISODate } from "@/utils/date";

export function useEvents({ startDate, endDate, eventType, subEvent }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!startDate || !endDate) return;
    (async () => {
      setLoading(true);
      setError("");
      try {
        const res = await getEvents({
          start_date: toISODate(startDate),
          end_date: toISODate(endDate),
          event_type: eventType || undefined,
          sub_event_type: subEvent || undefined
        });
        setData(Array.isArray(res) ? res : []);
      } catch (e) {
        setError(e?.message || "Failed to fetch events");
      } finally {
        setLoading(false);
      }
    })();
  }, [startDate, endDate, eventType, subEvent]);

  return { data, loading, error };
}
