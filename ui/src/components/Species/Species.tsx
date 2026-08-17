import { useEffect, useRef, useState } from "react";
import { confirmSpecies, fetchSpeciesNames, type Entry } from "../../lib/api";
import "./Species.css";

interface Props {
  entry: Entry;
  onUpdateMany: (updated: Entry[]) => void;
  onClose: () => void;
}

const MAX_MATCHES = 8;

export function Species({ entry, onUpdateMany, onClose }: Props) {
  const [names, setNames] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchSpeciesNames().then(setNames);
    inputRef.current?.focus();
  }, []);

  const matches = query
    ? names.filter((n) => n.toLowerCase().includes(query.toLowerCase())).slice(0, MAX_MATCHES)
    : [];

  const confirm = async (name: string | null) => {
    if (entry.burst == null) return;
    onUpdateMany(await confirmSpecies(entry.burst, name));
    onClose();
  };

  return (
    <div className="species" onPointerDown={(e) => e.stopPropagation()}>
      <div className="species-title">species · applies to whole burst</div>
      {(entry.species ?? []).map((p) => {
        const name = p.common ?? p.scientific;
        return (
          <button key={p.scientific} className="species-row" onClick={() => confirm(name)}>
            <span>
              {name}
              {entry.species_user === name && <span className="species-check"> ✓</span>}
            </span>
            <span className="species-conf">{p.confidence.toFixed(2)}</span>
          </button>
        );
      })}
      <input
        ref={inputRef}
        value={query}
        placeholder="other species…"
        spellCheck={false}
        onChange={(e) => {
          setQuery(e.target.value);
          setHighlight(0);
        }}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "ArrowDown") setHighlight((h) => Math.min(h + 1, matches.length - 1));
          else if (e.key === "ArrowUp") setHighlight((h) => Math.max(h - 1, 0));
          else if (e.key === "Enter" && matches[highlight]) confirm(matches[highlight]);
          else if (e.key === "Escape") onClose();
        }}
      />
      {matches.map((n, i) => (
        <button
          key={n}
          className={`species-row${i === highlight ? " species-row-active" : ""}`}
          onPointerEnter={() => setHighlight(i)}
          onClick={() => confirm(n)}
        >
          <span>
            {n}
            {entry.species_user === n && <span className="species-check"> ✓</span>}
          </span>
        </button>
      ))}
      {entry.species_user && (
        <button className="species-row species-clear" onClick={() => confirm(null)}>
          clear confirmation
        </button>
      )}
    </div>
  );
}
