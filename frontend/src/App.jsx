import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamAsk } from "./sse.js";

export default function App() {
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState("");
  const [settings, setSettings] = useState({ mode: "dense", k: 5, graph: false, model: "" });
  const [messages, setMessages] = useState([]); // {role:"user"|"assistant", ...}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((c) => {
        setConfig(c);
        setSettings((s) => ({ ...s, model: c.model, graph: c.graph }));
      })
      .catch(() => setConfigError("Не удаётся связаться с сервером. Запущен ли uvicorn api:app --port 8000?"));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Меняем последнее сообщение (ответ, который сейчас растёт).
  const patchLast = (fn) =>
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? { ...m, ...fn(m) } : m)));

  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((ms) => [...ms, { role: "user", text: question }, { role: "assistant", text: "", sources: [], waiting: true }]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamAsk(
        { question, k: settings.k, mode: settings.mode, graph: settings.graph, model: settings.model || null },
        (event, data) => {
          if (event === "sources") patchLast(() => ({ sources: data.sources }));
          else if (event === "token") patchLast((m) => ({ text: m.text + data.t, waiting: false }));
          else if (event === "refusal") patchLast(() => ({ refusal: data.message, waiting: false }));
          else if (event === "error") patchLast(() => ({ error: data.message, waiting: false }));
          else if (event === "done") patchLast(() => ({ meta: data, waiting: false }));
        },
        ctrl.signal
      );
    } catch (e) {
      if (e.name === "AbortError") patchLast(() => ({ stopped: true, waiting: false }));
      else patchLast(() => ({ error: String(e.message || e), waiting: false }));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));

  return (
    <div className="page">
      <header className="top">
        <h1>Ассистент по документации PyTorch</h1>
        <p>Вопросы на русском или английском. Ответ строится только по найденным фрагментам документации.</p>
      </header>

      <div className="layout">
        <main className="chat">
          {configError && <p className="notice error">{configError}</p>}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div className="msg user" key={i}>{m.text}</div>
            ) : (
              <Answer key={i} m={m} />
            )
          )}
          <div ref={endRef} />

          <form
            className="composer"
            onSubmit={(e) => { e.preventDefault(); ask(input); }}
          >
            <textarea
              value={input}
              rows={2}
              placeholder="Вопрос по документации PyTorch"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input); }
              }}
            />
            {busy ? (
              <button type="button" onClick={() => abortRef.current?.abort()}>Остановить</button>
            ) : (
              <button type="submit" className="primary" disabled={!input.trim()}>Спросить</button>
            )}
          </form>
        </main>

        <aside className="settings">
          <h2>Настройки</h2>
          <label>
            Режим поиска
            <select value={settings.mode} onChange={(e) => set("mode", e.target.value)}>
              {(config?.modes || ["dense", "hybrid"]).map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <label>
            Фрагментов в контексте: {settings.k}
            <input type="range" min="3" max="8" value={settings.k} onChange={(e) => set("k", +e.target.value)} />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={settings.graph}
              disabled={!config?.graph}
              onChange={(e) => set("graph", e.target.checked)}
            />
            Расширять контекст по графу связей
          </label>
          <label>
            Модель (Ollama)
            <input type="text" value={settings.model} onChange={(e) => set("model", e.target.value)} />
          </label>
          {config && <p className="small">Чанков в индексе: {config.chunks}</p>}
          <button onClick={() => setMessages([])} disabled={busy || !messages.length}>Очистить историю</button>
        </aside>
      </div>
    </div>
  );
}

function Answer({ m }) {
  const cited = new Set(m.meta?.cited || []);
  return (
    <div className="msg assistant">
      {m.waiting && <p className="muted">Ищу в документации…</p>}
      {m.refusal && <p className="refusal">{m.refusal}</p>}
      {m.text && (
        <div className="prose">
          <ReactMarkdown>{m.text}</ReactMarkdown>
        </div>
      )}
      {m.stopped && <p className="muted">Остановлено.</p>}
      {m.error && <p className="notice error">Ошибка: {m.error}</p>}
      {m.meta?.invented?.length > 0 && (
        <p className="notice">В ответе есть ссылки на несуществующие фрагменты: {m.meta.invented.join(", ")}.</p>
      )}
      {m.meta && !m.refusal && !m.error && m.meta.cited.length === 0 && (
        <p className="muted">В ответе нет ссылок на фрагменты: проверяй его по списку источников.</p>
      )}
      {m.sources?.length > 0 && (
        <details className="sources" open={false}>
          <summary>Источники ({m.sources.length})</summary>
          <ol>
            {m.sources.map((s) => (
              <li key={s.n} className={cited.has(s.n) ? "cited" : ""}>
                <span className="n">[{s.n}]</span> {s.title}
                {s.related && <span className="tag">по ссылкам</span>}
                <code>{s.file}</code>
              </li>
            ))}
          </ol>
        </details>
      )}
      {m.meta && <p className="muted small">{m.meta.seconds} с</p>}
    </div>
  );
}
