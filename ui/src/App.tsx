import { useCallback, useEffect, useRef, useState } from "react";
import {
  exportSidecars,
  fetchEntries,
  openFolder,
  subscribeEvents,
  type Entry,
  type ProgressEvent,
} from "./lib/api";
import { Settings } from "./components/Settings/Settings";
import { DEFAULT_SETTINGS, loadSettings, type PipelineSettings } from "./lib/settings";
import { Detail } from "./components/Detail/Detail";
import { Grid } from "./components/Grid/Grid";
import "./App.css";

export default function App() {
  const [path, setPath] = useState("data/labeling-set");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const current = selected !== null ? (entries.find((e) => e.id === selected) ?? null) : null;
  const gridScroll = useRef(0);
  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);
  const [exportMsg, setExportMsg] = useState<{ text: string; ok: boolean } | null>(null);

  useEffect(() => setSettings(loadSettings()), []);

  const doExport = useCallback(async () => {
    try {
      const r = await exportSidecars();
      setExportMsg({
        text: `✓ ${r.written} sidecars written${r.skipped_foreign ? `, ${r.skipped_foreign} foreign skipped` : ""}`,
        ok: true,
      });
      setTimeout(() => setExportMsg(null), 5000);
    } catch (e) {
      setExportMsg({ text: String(e), ok: false });
    }
  }, []);

  const open = useCallback(async () => {
    setError(null);
    setRunning(true);
    setEntries([]);
    try {
      await openFolder(path, settings);
      const stop = subscribeEvents(async (e) => {
        setProgress(e);
        if (e.type === "done") {
          setEntries(await fetchEntries());
          setRunning(false);
          stop();
        } else if (e.type === "error") {
          setError(e.message ?? "pipeline failed");
          setRunning(false);
          stop();
        }
      });
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }, [path, settings]);

  useEffect(() => {
    fetchEntries().then((existing) => {
      if (existing.length) setEntries(existing);
    });
  }, []);

  return (
    <div className="app">
      <header>
        <span className="logo">fovea</span>
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !running && open()}
          placeholder="folder of CR3s"
          spellCheck={false}
        />
        <button onClick={open} disabled={running}>
          {running ? "working…" : "open"}
        </button>
        {running && progress?.type === "progress" && (
          <span className="progress">
            {progress.stage} {progress.done}/{progress.total || "?"}
          </span>
        )}
        {!running && entries.length > 0 && (
          <span className="progress">{entries.length} frames</span>
        )}
        {error && <span className="error">{error}</span>}
        {exportMsg && (
          <span className={exportMsg.ok ? "export-ok" : "error"}>{exportMsg.text}</span>
        )}
        <span className="spacer" />
        <button onClick={doExport} disabled={running || entries.length === 0}>
          export
        </button>
        <Settings settings={settings} onChange={setSettings} />
      </header>
      {current ? (
        <Detail
          entry={current}
          entries={entries}
          onSelect={setSelected}
          onClose={() => setSelected(null)}
          onUpdate={(u) => setEntries((es) => es.map((e) => (e.id === u.id ? u : e)))}
          onUpdateMany={(us) =>
            setEntries((es) => {
              const byId = new Map(us.map((u) => [u.id, u]));
              return es.map((e) => byId.get(e.id) ?? e);
            })
          }
        />
      ) : (
        <Grid entries={entries} selected={selected} onSelect={setSelected} scrollPos={gridScroll} />
      )}
    </div>
  );
}
