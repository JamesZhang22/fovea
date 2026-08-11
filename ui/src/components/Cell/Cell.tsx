import { thumbUrl, type Entry } from "../../lib/api";
import "./Cell.css";

const IMG_H = 157;

export function Cell({
  entry,
  selected,
  onSelect,
  style,
}: {
  entry: Entry;
  selected: boolean;
  onSelect: (id: number) => void;
  style: React.CSSProperties;
}) {
  const best = entry.rank === 1 && entry.burst_size > 1;
  return (
    <div
      className={`cell${selected ? " cell-selected" : ""}`}
      style={style}
      onClick={() => onSelect(entry.id)}
    >
      <img src={thumbUrl(entry.id)} loading="lazy" height={IMG_H} alt={entry.name} />
      <div className="cell-bar">
        <span className="cell-name">{entry.name}</span>
        {best && <span className="badge badge-best">★</span>}
        {entry.burst_size > 1 && <span className="badge">{entry.burst_size}</span>}
        {entry.eye && entry.eye.confidence >= 0.5 && <span className="badge badge-eye">◉</span>}
      </div>
    </div>
  );
}
