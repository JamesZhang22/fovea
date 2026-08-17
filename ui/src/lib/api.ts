import type { components } from "./api-types";

export type Entry = components["schemas"]["Entry"];
export type Eye = components["schemas"]["Eye"];
export type BirdBox = components["schemas"]["BirdBox"];
export type AFPoint = components["schemas"]["AFPoint"];
export type SpeciesPrediction = components["schemas"]["SpeciesPrediction"];
export type SpeciesModelStatus = components["schemas"]["SpeciesModelStatus"];

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
  species?: boolean;
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

export async function confirmSpecies(burst: number, common: string | null): Promise<Entry[]> {
  const r = await fetch("/api/species", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ burst, common }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
  return r.json();
}

export async function fetchSpeciesModel(): Promise<SpeciesModelStatus> {
  return (await fetch("/api/species/model")).json();
}

export async function startSpeciesModelDownload(): Promise<SpeciesModelStatus> {
  return (await fetch("/api/species/model", { method: "POST" })).json();
}

let namesCache: string[] | null = null;
export async function fetchSpeciesNames(): Promise<string[]> {
  if (!namesCache) namesCache = await (await fetch("/api/species/names")).json();
  return namesCache!;
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
