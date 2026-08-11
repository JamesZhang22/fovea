import { useEffect, useRef, useState } from "react";
import type { Entry } from "../lib/api";
import { Cell } from "./Cell";

const CELL_W = 236;
const IMG_H = 157;
const CELL_H = IMG_H + 40;
const GAP = 8;
const OVERSCAN_ROWS = 3;

interface Props {
  entries: Entry[];
  selected: number | null;
  onSelect: (id: number) => void;
}

export function Grid({ entries, selected, onSelect }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ top: 0, height: 800, width: 1200 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () =>
      setViewport({ top: el.scrollTop, height: el.clientHeight, width: el.clientWidth });
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, []);

  const cols = Math.max(1, Math.floor((viewport.width - GAP) / (CELL_W + GAP)));
  const rows = Math.ceil(entries.length / cols);
  const firstRow = Math.max(0, Math.floor(viewport.top / (CELL_H + GAP)) - OVERSCAN_ROWS);
  const lastRow = Math.min(
    rows - 1,
    Math.ceil((viewport.top + viewport.height) / (CELL_H + GAP)) + OVERSCAN_ROWS,
  );

  const visible: { entry: Entry; row: number; col: number }[] = [];
  for (let row = firstRow; row <= lastRow; row++) {
    for (let col = 0; col < cols; col++) {
      const i = row * cols + col;
      if (i < entries.length) visible.push({ entry: entries[i], row, col });
    }
  }

  return (
    <div className="grid-scroll" ref={ref}>
      <div style={{ position: "relative", height: rows * (CELL_H + GAP) + GAP }}>
        {visible.map(({ entry, row, col }) => (
          <Cell
            key={entry.id}
            entry={entry}
            selected={entry.id === selected}
            onSelect={onSelect}
            style={{
              position: "absolute",
              top: GAP + row * (CELL_H + GAP),
              left: GAP + col * (CELL_W + GAP),
              width: CELL_W,
              height: CELL_H,
            }}
          />
        ))}
      </div>
    </div>
  );
}
