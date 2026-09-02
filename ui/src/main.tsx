import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Tooltip } from "radix-ui";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Tooltip.Provider delayDuration={400}>
      <App />
    </Tooltip.Provider>
  </StrictMode>,
);
