import type { Entry } from "../../lib/api";
import type { OverlayToggles } from "../Detail/Detail";
import "./Overlays.css";

/** Markers in full-resolution image coordinates, scaled by the parent transform. */
export function Overlays({ entry, toggles }: { entry: Entry; toggles: OverlayToggles }) {
  const W = entry.width ?? 6960;
  const H = entry.height ?? 4640;
  return (
    <svg className="overlays" viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      {toggles.af &&
        entry.af?.display_points.map((p, i) => (
          <rect
            key={i}
            x={p.cx - p.w / 2}
            y={p.cy - p.h / 2}
            width={p.w}
            height={p.h}
            className={p.in_focus ? "ov-af-focus" : "ov-af"}
          />
        ))}
      {toggles.eye && entry.eye && (
        <circle
          cx={entry.eye.x}
          cy={entry.eye.y}
          r={40}
          className={entry.eye_used ? "ov-eye" : "ov-eye-weak"}
        />
      )}
    </svg>
  );
}
