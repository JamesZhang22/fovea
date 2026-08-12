import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchEntries,
  openFolder,
  subscribeEvents,
  type Entry,
  type ProgressEvent,
} from "./lib/api";
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

  const open = useCallback(async () => {
    setError(null);
    setRunning(true);
    setEntries([]);
    try {
      await openFolder(path);
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
  }, [path]);

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
      </header>
      {current ? (
        <Detail
          entry={current}
          entries={entries}
          onSelect={setSelected}
          onClose={() => setSelected(null)}
          onUpdate={(u) => setEntries((es) => es.map((e) => (e.id === u.id ? u : e)))}
        />
      ) : (
        <Grid entries={entries} selected={selected} onSelect={setSelected} scrollPos={gridScroll} />
      )}
    </div>
  );
}
