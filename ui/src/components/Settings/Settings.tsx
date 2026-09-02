import { useEffect, useState } from "react";
import { Popover, Select, Switch, Tooltip } from "radix-ui";
import {
  fetchSpeciesModel,
  startSpeciesModelDownload,
  type SpeciesModelStatus,
} from "../../lib/api";
import { saveSettings, SPECIES_REGIONS, type PipelineSettings } from "../../lib/settings";
import "./Settings.css";

const DOWNLOAD_POLL_MS = 1000;
const ALL_REGIONS = "all"; // Select reserves the empty string, so null needs a sentinel.

const METRICS: [string, string][] = [
  ["brenner", "brenner"],
  ["tenengrad", "tenengrad"],
  ["edge_sharpness", "edge sharpness"],
];

interface Props {
  settings: PipelineSettings;
  onChange: (s: PipelineSettings) => void;
}

export function Settings({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState<SpeciesModelStatus | null>(null);

  useEffect(() => saveSettings(settings), [settings]);

  // model status on open, poll while a download runs.
  useEffect(() => {
    if (!open) return;
    fetchSpeciesModel().then(setModel);
  }, [open]);
  useEffect(() => {
    if (!open || !model?.downloading) return;
    const t = setTimeout(() => fetchSpeciesModel().then(setModel), DOWNLOAD_POLL_MS);
    return () => clearTimeout(t);
  }, [open, model]);

  const set = (patch: Partial<PipelineSettings>) => onChange({ ...settings, ...patch });

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <Popover.Trigger asChild>
            <button className="settings-toggle" aria-label="settings">
              ⚙
            </button>
          </Popover.Trigger>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="tooltip" sideOffset={6}>
            settings
            <Tooltip.Arrow className="tooltip-arrow" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>

      <Popover.Portal>
        <Popover.Content className="settings-panel" align="end" sideOffset={8}>
          <div className="settings-title">Pipeline</div>

          <div className="settings-row">
            <span>Bird detection</span>
            <Toggle
              checked={settings.detect}
              onChange={(v) =>
                set({ detect: v, eye: v && settings.eye, species: v && settings.species })
              }
            />
          </div>

          <Row label="Eye scoring" disabled={!settings.detect}>
            <Toggle
              checked={settings.eye}
              disabled={!settings.detect}
              onChange={(v) => set({ eye: v })}
            />
          </Row>

          <Row label="Species ID" disabled={!settings.detect}>
            {model && !model.present ? (
              <SpeciesDownload model={model} onStatus={setModel} />
            ) : (
              <Toggle
                checked={settings.species}
                disabled={!settings.detect}
                onChange={(v) => set({ species: v })}
              />
            )}
          </Row>

          <Row label="Region" disabled={!settings.detect || !settings.species}>
            <Dropdown
              value={settings.species_region ?? ALL_REGIONS}
              disabled={!settings.detect || !settings.species}
              options={[[ALL_REGIONS, "All regions"], ...SPECIES_REGIONS]}
              onChange={(v) => set({ species_region: v === ALL_REGIONS ? null : v })}
            />
          </Row>

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
            <Dropdown
              value={settings.metric}
              options={METRICS}
              onChange={(v) => set({ metric: v as PipelineSettings["metric"] })}
            />
          </div>

          <div className="settings-note">Applies on next folder open</div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function Row({
  label,
  disabled,
  children,
}: {
  label: string;
  disabled: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={`settings-row${disabled ? " settings-row-disabled" : ""}`}>
      <span>{label}</span>
      {children}
    </div>
  );
}

function Toggle({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Switch.Root
      className="switch"
      checked={checked}
      disabled={disabled}
      onCheckedChange={onChange}
    >
      <Switch.Thumb className="switch-knob" />
    </Switch.Root>
  );
}

function Dropdown({
  value,
  disabled,
  options,
  onChange,
}: {
  value: string;
  disabled?: boolean;
  options: [string, string][];
  onChange: (v: string) => void;
}) {
  return (
    <Select.Root value={value} disabled={disabled} onValueChange={onChange}>
      <Select.Trigger className="select-trigger" aria-label={value}>
        <Select.Value />
        <Select.Icon>▾</Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="select-content" position="popper" sideOffset={4}>
          <Select.Viewport>
            {options.map(([v, label]) => (
              <Select.Item key={v} className="select-item" value={v}>
                <Select.ItemText>{label}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
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
