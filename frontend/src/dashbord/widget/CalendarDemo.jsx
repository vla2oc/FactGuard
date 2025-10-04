import React from "react";
import { Calendar } from "@/components/ui/calendar";

export default function CalendarDemo({
  startDate,
  endDate,
  isEventDay,
  setStartDate,
  setEndDate,
}) {
  return (
    <div className="rounded-lg border shadow-sm bg-gray-900 text-white inline-block">
      <Calendar
        mode="range"
        selected={{ from: startDate, to: endDate }}
        onSelect={(r) => {
          if (r?.from) setStartDate(r.from);
          if (r?.to) setEndDate(r.to);
        }}
        numberOfMonths={2}
        captionLayout="dropdown"
        modifiers={{ hasEvent: (d) => isEventDay?.(d) }}
        modifiersClassNames={{
          hasEvent: "bg-gray-700 text-white rounded-md",
        }}
      />
    </div>
  );
}
