
"use client"

import * as React from "react"

import { Calendar } from "@/components/ui/calendar"


export default function CalendarDemo({
  startDate,
  endDate,
  isEventDay,
  setStartDate,
  setEndDate,
  minDate = new Date(2022, 1, 24), // прикинул минимум
  maxDate = new Date(),            // сегодня
}) {
  const selected = React.useMemo(
    () => ({ from: startDate ?? undefined, to: endDate ?? undefined }),
    [startDate, endDate]
  );

  const handleSelect = (range) => {
    if (!range) return;
    let { from, to } = range;

    // если кликнули один день — считаем диапазон одним днём
    if (from && !to) to = from;
    // гарантия порядка
    if (from && to && to < from) [from, to] = [to, from];

    if (from) setStartDate(from);
    if (to) setEndDate(to);
  };

  const fromYear = (minDate ?? new Date(2025, 0, 1)).getFullYear();
  const toYear = (maxDate ?? new Date()).getFullYear();

  return (
    <div className="rounded-lg border bg-stone-900">
      <Calendar
        mode="range"
        selected={selected}
        onSelect={handleSelect}
        numberOfMonths={2}
        captionLayout="dropdown"
        fromYear={fromYear}
        toYear={toYear}
        disabled={{ before: minDate, after: maxDate }}
        showOutsideDays
        modifiers={{ hasEvent: (d) => !!isEventDay?.(d) }}
        modifiersClassNames={{ hasEvent: "bg-gray-800 text-white rounded-md" }}
      />
    </div>
  );
}