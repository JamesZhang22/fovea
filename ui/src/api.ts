export interface Eye {
  x: number;
  y: number;
  confidence: number;
}

export interface BirdBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  confidence: number;
}

export interface AFPoint {
  cx: number;
  cy: number;
  w: number;
  h: number;
  in_focus: boolean;
  selected: boolean;
}

export interface Entry {
  id: number;
  name: string;
  width: number;
  height: number;
  orientation: number;
  burst: number;
  burst_size: number;
  rank: number | null;
  metrics: Record<string, number> | null;
  eye: Eye | null;
  birds: BirdBox[] | null;
  af: { lattice: boolean; display_points: AFPoint[] } | null;
  shot_time: string | null;
}

export interface ProgressEvent {
  type: "progress" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  count?: number;
  message?: string;
}

export async function openFolder(path: string): Promise<void> {
  const r = await fetch("/api/folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
}

export async function fetchEntries(): Promise<Entry[]> {
  const r = await fetch("/api/entries");
  return r.json();
}

export function thumbUrl(id: number, w = 400): string {
  return `/api/image/${id}/thumb?w=${w}`;
}

export function fullUrl(id: number): string {
  return `/api/image/${id}/full`;
}

export function subscribeEvents(onEvent: (e: ProgressEvent) => void): () => void {
  const source = new EventSource("/api/events");
  source.onmessage = (m) => onEvent(JSON.parse(m.data));
  source.onerror = () => source.close();
  return () => source.close();
}
