/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Gọi API bằng đường dẫn tương đối (/api, /health). Lúc dev, Vite chuyển tiếp sang FastAPI; sau này (GĐ10)
// reverse proxy làm đúng việc đó nên code React không đổi.
const backend = process.env.VITE_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": backend, "/health": backend },
  },
  preview: {
    port: 4173,
    proxy: { "/api": backend, "/health": backend },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
