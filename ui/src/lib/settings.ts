export interface PipelineSettings {
  detect: boolean;
  eye: boolean;
  species: boolean;
  species_region: string | null;
  gap_seconds: number;
  metric: "brenner" | "tenengrad" | "edge_sharpness";
}

export const DEFAULT_SETTINGS: PipelineSettings = {
  detect: true,
  eye: true,
  species: true,
  species_region: null,
  gap_seconds: 3.0,
  metric: "brenner",
};

// keys match core REGIONS in species/classify.py.
export const SPECIES_REGIONS: [string, string][] = [
  ["north-america", "North America"],
  ["south-america", "South America"],
  ["eurasia", "Eurasia"],
  ["africa", "Africa"],
  ["south-asia", "South & SE Asia"],
  ["australasia", "Australasia"],
];

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
