import type { components } from "./api-types";

export type Entry = components["schemas"]["Entry"];
export type Eye = components["schemas"]["Eye"];
export type BirdBox = components["schemas"]["BirdBox"];
export type AFPoint = components["schemas"]["AFPoint"];

// SSE events are outside the OpenAPI schema, mirrored from api/session.py
export interface ProgressEvent {
  type: "progress" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  count?: number;
  message?: string;
}

export interface OpenOptions {
  detect?: boolean;
  eye?: boolean;
  gap_seconds?: number;
  metric?: string;
}

export async function openFolder(path: string, options: OpenOptions = {}): Promise<void> {
  const r = await fetch("/api/folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, ...options }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
}

export async function exportSidecars(): Promise<{ written: number; skipped_foreign: number }> {
  const r = await fetch("/api/export", { method: "POST" });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
  return r.json();
}

export async function rate(
  id: number,
  patch: { rating?: number; rejected?: boolean },
): Promise<Entry> {
  const r = await fetch("/api/rate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, ...patch }),
  });
  return r.json();
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
