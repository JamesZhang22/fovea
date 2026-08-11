import { useCallback, useEffect, useRef, useState } from "react";
import { fullUrl, type Entry } from "../../lib/api";
import { Filmstrip } from "../Filmstrip/Filmstrip";
import { Overlays } from "../Overlays/Overlays";
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
}

const MAX_SCALE = 4;
const CLICK_SLOP_PX = 5; // pointer travel below this counts as a click, not a drag

interface Transform {
  scale: number;
  x: number;
  y: number;
}

export function Detail({ entry, entries, onSelect, onClose }: Props) {
  const viewRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ w: 800, h: 600 });
  const [transform, setTransform] = useState<Transform | null>(null); // null = fit, centered
  const [toggles, setToggles] = useState<OverlayToggles>({ eye: true, af: true, info: true });
  const [loadedId, setLoadedId] = useState<number | null>(null);

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
      if (e.key === "Escape") onClose();
      else if (e.key === "e") setToggles((t) => ({ ...t, eye: !t.eye }));
      else if (e.key === "a") setToggles((t) => ({ ...t, af: !t.af }));
      else if (e.key === "i") setToggles((t) => ({ ...t, info: !t.info }));
      else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const i = entries.findIndex((b) => b.id === entry.id);
        const next = entries[i + (e.key === "ArrowRight" ? 1 : -1)];
        if (next) onSelect(next.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [entry.id, entries, onClose, onSelect, clampPos, pos.x, pos.y, scale, view]);

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
    m && m.anisotropy !== undefined
      ? m.anisotropy > 0.3
        ? `motion ${m.motion_angle}°`
        : "defocus-dominant"
      : null;
  const zoomLabel = scale <= fitScale * 1.01 ? "fit" : `${Math.round(scale * 100)}%`;
  return (
    <div className="infobar">
      <span>{entry.name}</span>
      {entry.rank !== null && <span>rank {(entry.rank * 100).toFixed(0)}%</span>}
      {m && <span>focus {m.brenner.toFixed(0)}</span>}
      {blur && <span>{blur}</span>}
      {entry.eye && <span>eye {entry.eye.confidence.toFixed(2)}</span>}
      <span className="dim">{zoomLabel} · click zoom · pinch/scroll · e/a/i overlays · esc</span>
    </div>
  );
}
