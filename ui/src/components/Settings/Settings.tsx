import { useEffect, useRef, useState } from "react";
import {
  fetchSpeciesModel,
  startSpeciesModelDownload,
  type SpeciesModelStatus,
} from "../../lib/api";
import { saveSettings, SPECIES_REGIONS, type PipelineSettings } from "../../lib/settings";
import "./Settings.css";

const DOWNLOAD_POLL_MS = 1000;

interface Props {
  settings: PipelineSettings;
  onChange: (s: PipelineSettings) => void;
}

export function Settings({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState<SpeciesModelStatus | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => saveSettings(settings), [settings]);

  // model status on open, poll while a download runs
  useEffect(() => {
    if (!open) return;
    fetchSpeciesModel().then(setModel);
  }, [open]);
  useEffect(() => {
    if (!open || !model?.downloading) return;
    const t = setTimeout(() => fetchSpeciesModel().then(setModel), DOWNLOAD_POLL_MS);
    return () => clearTimeout(t);
  }, [open, model]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  const set = (patch: Partial<PipelineSettings>) => onChange({ ...settings, ...patch });

  return (
    <div className="settings" ref={ref}>
      <button
        className={`settings-toggle${open ? " settings-toggle-open" : ""}`}
        onClick={() => setOpen((o) => !o)}
        title="settings"
      >
        ⚙
      </button>
      {open && (
        <div className="settings-panel">
          <div className="settings-title">Pipeline</div>

          <div className="settings-row">
            <span>Bird detection</span>
            <Switch
              checked={settings.detect}
              onChange={(v) =>
                set({ detect: v, eye: v && settings.eye, species: v && settings.species })
              }
            />
          </div>

          <div className={`settings-row${settings.detect ? "" : " settings-row-disabled"}`}>
            <span>Eye scoring</span>
            <Switch
              checked={settings.eye}
              disabled={!settings.detect}
              onChange={(v) => set({ eye: v })}
            />
          </div>

          <div className={`settings-row${settings.detect ? "" : " settings-row-disabled"}`}>
            <span>Species ID</span>
            {model && !model.present ? (
              <SpeciesDownload model={model} onStatus={setModel} />
            ) : (
              <Switch
                checked={settings.species}
                disabled={!settings.detect}
                onChange={(v) => set({ species: v })}
              />
            )}
          </div>

          <div
            className={`settings-row${settings.detect && settings.species ? "" : " settings-row-disabled"}`}
          >
            <span>Region</span>
            <select
              value={settings.species_region ?? ""}
              disabled={!settings.detect || !settings.species}
              onChange={(e) => set({ species_region: e.target.value || null })}
            >
              <option value="">All regions</option>
              {SPECIES_REGIONS.map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="settings-row">
            <span>Burst gap</span>
            <span className="settings-control">
              <input
                type="number"
                min={0.5}
                max={30}
                step={0.5}
                value={settings.gap_seconds}
                onChange={(e) => set({ gap_seconds: Number(e.target.value) })}
              />
              <span className="settings-unit">s</span>
            </span>
          </div>

          <div className="settings-row">
            <span>Rank metric</span>
            <select
              value={settings.metric}
              onChange={(e) => set({ metric: e.target.value as PipelineSettings["metric"] })}
            >
              <option value="brenner">brenner</option>
              <option value="tenengrad">tenengrad</option>
              <option value="edge_sharpness">edge sharpness</option>
            </select>
          </div>

          <div className="settings-note">Applies on next folder open</div>
        </div>
      )}
    </div>
  );
}

function SpeciesDownload({
  model,
  onStatus,
}: {
  model: SpeciesModelStatus;
  onStatus: (s: SpeciesModelStatus) => void;
}) {
  if (model.downloading) {
    const pct = model.total_bytes ? Math.round((100 * model.done_bytes) / model.total_bytes) : 0;
    return <span className="settings-download-progress">downloading… {pct}%</span>;
  }
  return (
    <span className="settings-control">
      {model.error && <span className="settings-download-error">failed</span>}
      <button
        className="settings-download"
        title={model.error ?? undefined}
        onClick={() => startSpeciesModelDownload().then(onStatus)}
      >
        {model.error ? "retry" : "download model · 1.2 GB"}
      </button>
    </span>
  );
}

function Switch({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      className={`switch${checked ? " switch-on" : ""}`}
      disabled={disabled}
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
    >
      <span className="switch-knob" />
    </button>
  );
}
