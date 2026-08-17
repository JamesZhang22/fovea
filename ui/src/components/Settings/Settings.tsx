import { useEffect, useRef, useState } from "react";
import { saveSettings, type PipelineSettings } from "../../lib/settings";
import "./Settings.css";

interface Props {
  settings: PipelineSettings;
  onChange: (s: PipelineSettings) => void;
}

export function Settings({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => saveSettings(settings), [settings]);

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
            <Switch
              checked={settings.species}
              disabled={!settings.detect}
              onChange={(v) => set({ species: v })}
            />
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
