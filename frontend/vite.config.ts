import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND = "http://127.0.0.1:8848";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // SSE 必须关掉缓冲，否则事件会被攒着一次性吐出来
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        configure: (proxy: any) => {
          proxy.on("proxyRes", (proxyRes: any) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
