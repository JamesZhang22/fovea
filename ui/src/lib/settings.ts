export interface PipelineSettings {
  detect: boolean;
  eye: boolean;
  gap_seconds: number;
  metric: "brenner" | "tenengrad" | "edge_sharpness";
}

export const DEFAULT_SETTINGS: PipelineSettings = {
  detect: true,
  eye: true,
  gap_seconds: 3.0,
  metric: "brenner",
};

const STORAGE_KEY = "fovea-settings";

export function loadSettings(): PipelineSettings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}") };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: PipelineSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}
