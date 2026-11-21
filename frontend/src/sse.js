// Разбор потока Server-Sent Events.
// Сообщение выглядит так:  "event: token\ndata: {...}\n\n"  (пустая строка завершает сообщение).
// Чанки сети режут поток где угодно, поэтому копим буфер и разбираем только законченные сообщения.

export function createParser(onEvent) {
  let buf = "";
  return (chunk) => {
    buf += chunk.replace(/\r\n/g, "\n");
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  };
}

// EventSource умеет только GET, а нам нужно отправить JSON в теле, поэтому fetch + чтение потока.
export async function streamAsk(body, onEvent, signal) {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) throw new Error(`Сервер ответил ${resp.status}`);
  const feed = createParser(onEvent);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    feed(decoder.decode(value, { stream: true }));
  }
}
