import { useEffect, useState } from "react";
import { api as defaultApi } from "./api/client";
import type { Api } from "./api/types";
import { useReady, type ReadyState } from "./hooks/useReady";
import { AskPage } from "./pages/AskPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { ReadPage } from "./pages/ReadPage";

type Tab = "ask" | "read" | "benchmark";
const tabFromHash = (): Tab => (window.location.hash === "#benchmark" ? "benchmark" : window.location.hash === "#read" ? "read" : "ask");

export default function App({ api = defaultApi }: { api?: Api }) {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const ready = useReady(api);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="top">
        <h1>GermanQA</h1>
        <nav aria-label="Trang">
          <a href="#ask" aria-current={tab === "ask" ? "page" : undefined}>Hỏi đáp</a>
          <a href="#read" aria-current={tab === "read" ? "page" : undefined}>Đọc đoạn văn</a>
          <a href="#benchmark" aria-current={tab === "benchmark" ? "page" : undefined}>Số đo</a>
        </nav>
        <Status ready={ready} />
      </header>
      <main>
        {tab === "ask" && <AskPage ready={ready} api={api} />}
        {tab === "read" && <ReadPage ready={ready} api={api} />}
        {tab === "benchmark" && <BenchmarkPage api={api} />}
      </main>
    </div>
  );
}

function Status({ ready }: { ready: ReadyState }) {
  const map: Record<ReadyState["phase"], [string, string]> = {
    checking: ["dim", "Đang kiểm tra…"],
    loading: ["warn", "Đang nạp mô hình"],
    ready: ["ok", "Sẵn sàng"],
    failed: ["bad", "Lỗi nạp"],
    unreachable: ["bad", "Mất kết nối"],
  };
  const [cls, text] = map[ready.phase];
  const info = ready.phase === "ready" ? ready.info : null;
  return (
    <span className={`status ${cls}`} title={info ? `${info.model} · corpus ${info.corpus}` : undefined}>
      {text}
    </span>
  );
}
