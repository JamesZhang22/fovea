import { useEffect, useMemo, useRef } from "react";
import { thumbUrl, type Entry } from "../../lib/api";
import "./Filmstrip.css";

interface Props {
  entries: Entry[];
  currentId: number;
  onSelect: (id: number) => void;
}

/** All frames in shoot order, consecutive burst frames visually grouped. */
export function Filmstrip({ entries, currentId, onSelect }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  const groups = useMemo(() => {
    const out: Entry[][] = [];
    for (const e of entries) {
      const last = out[out.length - 1];
      if (last && last[0].burst === e.burst) last.push(e);
      else out.push([e]);
    }
    return out;
  }, [entries]);

  // center once on entry from the grid, afterwards only nudge frames back into view
  useEffect(() => {
    ref.current
      ?.querySelector(".strip-current")
      ?.scrollIntoView({ inline: "center", block: "nearest" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    ref.current
      ?.querySelector(".strip-current")
      ?.scrollIntoView({ inline: "nearest", block: "nearest" });
  }, [currentId]);

  return (
    <div className="filmstrip" ref={ref}>
      {groups.map((group) => (
        <div
          key={group[0].id}
          className={`strip-burst${group.length > 1 ? " strip-burst-multi" : ""}`}
        >
          {group.map((e) => (
            <div
              key={e.id}
              className={`strip-cell${e.id === currentId ? " strip-current" : ""}`}
              onClick={() => onSelect(e.id)}
            >
              <img src={thumbUrl(e.id, 200)} loading="lazy" alt={e.name} />
              {e.rank === 1 && group.length > 1 && <span className="strip-star">★</span>}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
