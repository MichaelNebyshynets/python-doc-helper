import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В режиме разработки запросы /api/... пересылаются на FastAPI (порт 8000),
// поэтому в коде фронтенда можно писать просто fetch("/api/ask").
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
