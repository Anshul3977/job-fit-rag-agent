import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const api = "http://127.0.0.1:8741";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/health": api,
      "/ready": api,
      "/samples": api,
      "/ingest": api,
      "/fit": api,
      "/runs": api,
      "/jobs": api,
      "/resumes": api,
      "/docs": api,
      "/openapi.json": api,
    },
  },
});
