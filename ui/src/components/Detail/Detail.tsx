import { useCallback, useEffect, useRef, useState } from "react";
import { fullUrl, rate, type Entry } from "../../lib/api";
import { Filmstrip } from "../Filmstrip/Filmstrip";
import { Overlays } from "../Overlays/Overlays";
import { Species } from "../Species/Species";
import "./Detail.css";

export interface OverlayToggles {
  eye: boolean;
  af: boolean;
  info: boolean;
}

interface Props {
  entry: Entry;
  entries: Entry[];
  onSelect: (id: number) => void;
  onClose: () => void;
  onUpdate: (updated: Entry) => void;
  onUpdateMany: (updated: Entry[]) => void;
}

const MAX_SCALE = 4;
// hotkeys must not fire while a field or a portalled Radix panel has focus
const HOTKEY_BLOCKERS =
  "input, select, textarea, [contenteditable], [role='dialog'], [role='menu'], [role='listbox']";
const CLICK_SLOP_PX = 5; // pointer travel below this counts as a click, not a drag

interface Transform {
  scale: number;
  x: number;
  y: number;
}

export function Detail({ entry, entries, onSelect, onClose, onUpdate, onUpdateMany }: Props) {
  const viewRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ w: 800, h: 600 });
  const [transform, setTransform] = useState<Transform | null>(null); // null = fit, centered
  const [toggles, setToggles] = useState<OverlayToggles>({ eye: true, af: true, info: true });
  const [loadedId, setLoadedId] = useState<number | null>(null);
  const [speciesOpen, setSpeciesOpen] = useState(false);

  const W = entry.width ?? 6960;
  const H = entry.height ?? 4640;
  const fitScale = Math.min(view.w / W, view.h / H, 1);
  const scale = transform?.scale ?? fitScale;
  const pos = transform ?? { x: (view.w - W * fitScale) / 2, y: (view.h - H * fitScale) / 2 };

  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setView({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // swap frames only after full decode, a half-decoded 6960px JPEG paints torn
  useEffect(() => {
    let alive = true;
    const im = new Image();
    im.src = fullUrl(entry.id);
    im.decode()
      .catch(() => undefined)
      .then(() => alive && setLoadedId(entry.id));
    return () => {
      alive = false;
    };
  }, [entry.id]);

  // reset to fit when the frame changes
  useEffect(() => setTransform(null), [entry.id]);

  const clampPos = useCallback(
    (x: number, y: number, s: number) => ({
      x: W * s <= view.w ? (view.w - W * s) / 2 : Math.min(0, Math.max(view.w - W * s, x)),
      y: H * s <= view.h ? (view.h - H * s) / 2 : Math.min(0, Math.max(view.h - H * s, y)),
    }),
    [W, H, view],
  );

  const zoomAt = useCallback(
    (viewX: number, viewY: number, targetScale: number) => {
      const s = Math.min(Math.max(targetScale, fitScale), MAX_SCALE);
      const ix = (viewX - pos.x) / scale;
      const iy = (viewY - pos.y) / scale;
      setTransform({ scale: s, ...clampPos(viewX - ix * s, viewY - iy * s, s) });
    },
    [fitScale, pos.x, pos.y, scale, clampPos],
  );

  // wheel must be non-passive to preventDefault, React's synthetic handler is passive
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      if (e.ctrlKey) {
        zoomAt(e.clientX - r.left, e.clientY - r.top, scale * Math.exp(-e.deltaY * 0.01));
      } else {
        setTransform({
          scale,
          ...clampPos(pos.x - e.deltaX, pos.y - e.deltaY, scale),
        });
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [scale, pos.x, pos.y, zoomAt, clampPos]);

  const drag = useRef<{
    startX: number;
    startY: number;
    baseX: number;
    baseY: number;
    moved: boolean;
  } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest?.(HOTKEY_BLOCKERS)) return;
      const i = entries.findIndex((b) => b.id === entry.id);
      const step = (delta: number) => {
        const next = entries[i + delta];
        if (next) onSelect(next.id);
      };
      if (e.key === "Escape") onClose();
      else if (e.key === "s" && entry.species) {
        e.preventDefault();
        setSpeciesOpen((o) => !o);
      } else if (e.key === "e") setToggles((t) => ({ ...t, eye: !t.eye }));
      else if (e.key === "a") setToggles((t) => ({ ...t, af: !t.af }));
      else if (e.key === "i") setToggles((t) => ({ ...t, info: !t.info }));
      else if (e.key === "ArrowLeft" || e.key === "j") step(-1);
      else if (e.key === "ArrowRight" || e.key === "k") step(1);
      else if (e.key >= "1" && e.key <= "5") {
        const n = Number(e.key);
        rate(entry.id, { rating: n === entry.user_rating ? 0 : n }).then(onUpdate);
      } else if (e.key === "x") {
        rate(entry.id, { rejected: !entry.rejected }).then((u) => {
          onUpdate(u);
          step(1);
        });
      } else if (e.key === "[" || e.key === "]") {
        const dir = e.key === "]" ? 1 : -1;
        let n = i + dir;
        while (n >= 0 && n < entries.length && entries[n].burst === entry.burst) n += dir;
        if (dir === -1 && n >= 0) {
          const b = entries[n].burst;
          while (n > 0 && entries[n - 1].burst === b) n -= 1;
        }
        if (n >= 0 && n < entries.length) onSelect(entries[n].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [entry, entries, onClose, onSelect, onUpdate]);

  // preload neighbours so filmstrip flips feel instant
  useEffect(() => {
    const i = entries.findIndex((b) => b.id === entry.id);
    for (const n of [entries[i - 1], entries[i + 1]]) {
      if (n) new Image().src = fullUrl(n.id);
    }
  }, [entry.id, entries]);

  const zoomedIn = scale > fitScale * 1.01;
  const showCurrent = loadedId === entry.id;

  return (
    <div className="detail">
      <div
        className={`loupe${zoomedIn ? " loupe-zoomed" : ""}`}
        ref={viewRef}
        onPointerDown={(e) => {
          drag.current = {
            startX: e.clientX,
            startY: e.clientY,
            baseX: pos.x,
            baseY: pos.y,
            moved: false,
          };
          (e.target as Element).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const dx = e.clientX - d.startX;
          const dy = e.clientY - d.startY;
          if (Math.abs(dx) + Math.abs(dy) > CLICK_SLOP_PX) d.moved = true;
          if (d.moved && zoomedIn) {
            setTransform({ scale, ...clampPos(d.baseX + dx, d.baseY + dy, scale) });
          }
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d || d.moved) return;
          const r = viewRef.current!.getBoundingClientRect();
          const vx = e.clientX - r.left;
          const vy = e.clientY - r.top;
          if (zoomedIn) {
            setTransform(null);
          } else {
            // zoom to 1:1 with the clicked image point centered
            const ix = (vx - pos.x) / scale;
            const iy = (vy - pos.y) / scale;
            setTransform({ scale: 1, ...clampPos(view.w / 2 - ix, view.h / 2 - iy, 1) });
          }
        }}
      >
        <div
          className="loupe-inner"
          style={{ transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})` }}
        >
          {loadedId !== null && (
            <img src={fullUrl(loadedId)} width={W} height={H} alt={entry.name} draggable={false} />
          )}
          {showCurrent && <Overlays entry={entry} toggles={toggles} />}
        </div>
        {toggles.info && <InfoBar entry={entry} scale={scale} fitScale={fitScale} />}
        {speciesOpen && (
          <Species
            entry={entry}
            onUpdateMany={onUpdateMany}
            onClose={() => setSpeciesOpen(false)}
          />
        )}
      </div>
      <Filmstrip entries={entries} currentId={entry.id} onSelect={onSelect} />
    </div>
  );
}

function InfoBar({
  entry,
  scale,
  fitScale,
}: {
  entry: Entry;
  scale: number;
  fitScale: number;
}) {
  const m = entry.metrics;
  const blur =
    m && m.anisotropy != null
      ? m.anisotropy > 0.3
        ? `motion ${m.motion_angle}°`
        : "defocus-dominant"
      : null;
  const zoomLabel = scale <= fitScale * 1.01 ? "fit" : `${Math.round(scale * 100)}%`;
  return (
    <div className="infobar">
      <span>{entry.name}</span>
      {entry.rank !== null && <span>rank {(entry.rank * 100).toFixed(0)}%</span>}
      {m && m.focus_score != null && (
        <span className="ib-score">score {m.focus_score.toFixed(0)}</span>
      )}
      {m && m.focus_score == null && m.focus_confidence != null && (
        <span className="ib-abstain">score —</span>
      )}
      {m && m.focus_radius_px != null && <span>blur {m.focus_radius_px.toFixed(1)}px</span>}
      {m && m.focus_percentile != null && (
        <span>top {Math.max(1, Math.round((1 - m.focus_percentile) * 100))}%</span>
      )}
      {m && <span>focus {m.brenner?.toFixed(0)}</span>}
      {blur && <span>{blur}</span>}
      {entry.species_user ? (
        <span className="ib-species ib-species-confirmed">{entry.species_user} ✓</span>
      ) : (
        entry.species?.[0] && (
          <span className="ib-species">
            {entry.species[0].common ?? entry.species[0].scientific}{" "}
            {entry.species[0].confidence.toFixed(2)}
          </span>
        )
      )}
      {entry.eye && <span>eye {entry.eye.confidence.toFixed(2)}</span>}
      {entry.user_rating != null && <span className="ib-rating">{entry.user_rating}★</span>}
      {entry.rejected && <span className="ib-reject">REJECTED</span>}
      <span className="dim">
        {zoomLabel} · j/k nav · 1-5 rate · x reject · [ ] burst · s species · e/a/i overlays · esc
      </span>
    </div>
  );
}
