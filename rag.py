#!/usr/bin/env python3
"""RAG-ассистент по документации PyTorch (RU/EN), всё локально через Ollama.

Запуск (из папки проекта):
    python rag.py ask "how does DataLoader num_workers work"   # один вопрос в консоли
    docker compose up -d                                       # запустить сервер Chroma (один раз после включения Docker)
    python rag.py index                                        # заполнить Chroma из chunks_full.json и emb.npy
    python rag.py eval                                         # поиск и отказ на eval.json и eval_holdout.json, граф на eval_graph.json
    python rag.py graph                                        # пересобрать граф связей graph.json
Веб-интерфейс: uvicorn api:app --port 8000 (см. README.md).

Нужны: запущенная Ollama (модели bge-m3 и модель для ответов), data/chunks_full.json и сервер Chroma
в Docker (docker compose up -d), заполненный командой python rag.py index (ей нужен emb.npy).
Схема: проверка имён torch.* -> привязка чанков с точным именем -> dense-поиск в Chroma (эмбеддинги bge-m3) ->
реранкер (bge-reranker-v2-m3) решает, отвечать ли -> модель генерирует ответ по найденным фрагментам.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"               # корпус, эмбеддинги, граф и тестовые вопросы
CHUNKS_PATH = DATA / "chunks_full.json"
EMB_PATH = DATA / "emb.npy"        # эмбеддинги чанков; нужны только для построения базы (python rag.py index)
CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")   # сервер Chroma из docker-compose.yml
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
COLLECTION = "pytorch_docs"

OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen2.5:7b"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

THRESH = 0.15     # порог оценки реранкера: ниже него отвечать не будем
GATE_POOL = 30    # сколько dense-кандидатов смотрит реранкер при решении об отказе

DOCS_PATH = DATA / "docs.json"     # docstring'и из пакета torch (получены в corpus_clean.ipynb)
GRAPH_PATH = DATA / "graph.json"   # граф связей между сущностями API (python rag.py graph)
GRAPH_TOP, GRAPH_PER_CHUNK, GRAPH_MAX_INDEG = 5, 3, 20   # расширение контекста по графу

SYSTEM = (
    "You are a PyTorch documentation assistant. Answer ONLY from the numbered excerpts provided. "
    "Do not use knowledge outside the excerpts and do not write code that is not in the excerpts. "
    "Take default values, numbers and API names verbatim from the excerpts. "
    "If the excerpts do not answer the question, say so and stop. "
    "Ignore excerpts that are not relevant to the question. "
    "End every sentence that uses an excerpt with its number in square brackets, e.g. [2]. "
    "Do not include URLs that are not in the excerpts."
)


# ----------------------------------------------------------------------------- утилиты

def tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def answer_language(query: str) -> str:
    return "Russian" if re.search(r"[а-яА-ЯёЁ]", query) else "English"


def is_hit(title: str, gold: list[str]) -> bool:
    """Чанк подходит, если его заголовок равен gold, начинается с него или содержит его как раздел."""
    for g in gold:
        if (title == g or title.startswith(g + " >")
                or title.endswith(" > " + g) or (" > " + g + " >") in title):
            return True
    return False


def build_context(res, max_chars: int = 3000) -> str:
    parts = []
    for i, (_score, c) in enumerate(res, 1):
        parts.append("[" + str(i) + "] " + c["text"][:max_chars])
    return "\n\n".join(parts)


def refusal_message(query: str, reason: str) -> str:
    ru = answer_language(query) == "Russian"
    if reason.startswith("в документации нет имени"):
        name = reason.split("имени", 1)[1].strip()
        return (f"Я не могу ответить: в документации нет имени {name}." if ru
                else f"I can't answer this: the documentation has no entity {name}.")
    return ("Я не нашёл в документации подходящего раздела по этому вопросу." if ru
            else "I couldn't find a relevant section of the documentation for this question.")


def check_citations(text: str, n_sources: int):
    """Номера ссылок [1], [1, 4] из ответа. Возвращает (все номера, номера без такого фрагмента)."""
    nums = []
    for grp in re.findall(r"\[([\d,\s]+)\]", text):
        nums += [int(x) for x in re.findall(r"\d+", grp)]
    cited = sorted(set(nums))
    return cited, [n for n in cited if n < 1 or n > n_sources]


# ----------------------------------------------------------------------------- векторная база

def get_db():
    """Клиент Chroma: подключается по HTTP к серверу, который запущен в Docker (docker compose up -d)."""
    import chromadb
    from chromadb.config import Settings
    # локальный сервер не должен ходить через системный прокси (иначе "Server disconnected without sending a response")
    for var in ("NO_PROXY", "no_proxy"):
        hosts = [h for h in os.environ.get(var, "").split(",") if h]
        os.environ[var] = ",".join(hosts + [h for h in (CHROMA_HOST, "localhost", "127.0.0.1") if h not in hosts])
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(anonymized_telemetry=False))
        client.heartbeat()
    except Exception as e:
        raise ConnectionError(f"не удаётся подключиться к Chroma ({CHROMA_HOST}:{CHROMA_PORT}): "
                              "запусти сервер командой  docker compose up -d  (Docker Desktop должен быть открыт)") from e
    return client


def build_index(chunks, emb, batch: int = 1000):
    """Создаёт коллекцию заново: id = номер чанка, вектор = готовый эмбеддинг, текст и заголовок в метаданных.

    Метрика cosine: Chroma возвращает расстояние = 1 - косинусная близость.
    Эмбеддинги передаём сами, поэтому Chroma не запускает свою модель эмбеддингов.
    """
    client = get_db()
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # коллекции ещё не было
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(ids=[str(c["id"]) for c in part],
                embeddings=emb[i:i + batch].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[{"file": c["file"], "kind": c["kind"], "title": c["title"]} for c in part])
    return col


# ----------------------------------------------------------------------------- граф

REF_RE = re.compile(r":(func|class|meth|mod|attr):`([^`]+)`")


def clean_ref(raw: str) -> str:
    """Имя из rst-ссылки: `~torch.nn.Module` -> torch.nn.Module, `текст <torch.add>` -> torch.add, `pad()` -> pad."""
    m = re.search(r"<([^>]+)>", raw)
    if m:
        raw = m.group(1)
    raw = raw.strip().lstrip("~.")
    if raw.endswith("()"):
        raw = raw[:-2]
    return raw


def node_name(c) -> str:
    """Имя узла графа для чанка: заголовок API-чанка или последняя часть заголовка текстового."""
    return c["title"] if c["kind"] == "api" else c["title"].split(" > ")[-1].strip()


def build_graph(chunks, docs):
    """Рёбра «A ссылается на B» из ссылок :func:, :class:, :meth:, :mod: в docstring'ах.

    Ссылки :attr: пропускаются: это в основном имена аргументов (other, alpha), а не API.
    Узлы: API-чанки (по заголовку) и текстовые чанки, заголовок которых заканчивается именем torch.xxx.
    Возвращает (граф {"edges": [...], "nodes": {имя: [id чанков]}}, статистика по ролям).
    """
    nodes = defaultdict(list)
    for c in chunks:
        if c["kind"] == "api":
            nodes[c["title"]].append(c["id"])
        else:
            last = c["title"].split(" > ")[-1].strip()
            if re.fullmatch(r"torch(\.\w+)+", last):
                nodes[last].append(c["id"])
    titles = set(nodes)
    tail_index = defaultdict(list)
    for t in titles:
        tail_index[t.split(".")[-1]].append(t)

    def resolve(src, name):
        if name in titles:
            return name
        parts = src.split(".")
        cand = ".".join(parts[:-1] + [name])        # сосед в том же классе или модуле (view -> Tensor.reshape)
        if cand in titles:
            return cand
        if "torch." + name in titles:
            return "torch." + name
        cand = ".".join(parts[:-2] + [name])        # на уровень выше
        if cand in titles:
            return cand
        if "." not in name:                          # короткое имя: берём, только если в корпусе оно одно
            c = tail_index.get(name, [])
            if len(c) == 1:
                return c[0]
        return None

    edges = set()
    stats = defaultdict(lambda: [0, 0])              # роль -> [ссылок, разрешено]
    for src, doc in docs.items():
        for role, raw in REF_RE.findall(doc):
            if role == "attr":
                continue
            dst = resolve(src, clean_ref(raw))
            stats[role][0] += 1
            if dst:
                stats[role][1] += 1
                if dst != src:
                    edges.add((src, dst))
    return {"edges": sorted(edges), "nodes": dict(nodes)}, dict(stats)


def covered(item, res) -> float:
    """Доля нужных узлов (gold) вопроса, которые попали в контекст."""
    got = sum(any(is_hit(c["title"], [g]) for _s, c in res) for g in item["gold"])
    return got / len(item["gold"])


# ----------------------------------------------------------------------------- ядро

class RAG:
    def __init__(self, gen_model: str = GEN_MODEL, thresh: float = THRESH,
                 client=None, rerank_device: str | None = None, collection=None):
        if client is None:
            import ollama
            # trust_env=False: запросы к локальной Ollama не должны идти через системный прокси
            client = ollama.Client(host=OLLAMA_HOST, trust_env=False)
        self.client = client
        self.gen_model = gen_model
        self.thresh = thresh
        self.rerank_device = rerank_device

        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)
        if collection is None:
            db = get_db()
            try:
                collection = db.get_collection(COLLECTION)
            except Exception as e:
                raise FileNotFoundError("в Chroma нет коллекции: запусти  python rag.py index") from e
        self.col = collection
        if self.col.count() != len(self.chunks):
            raise ValueError(
                f"в базе {self.col.count()} векторов, в chunks_full.json {len(self.chunks)} чанков: "
                "пересобери базу командой  python rag.py index")

        self._reranker = None
        self._bm25 = None
        self._graph = None
        self.title_index = defaultdict(list)
        for c in self.chunks:
            self.title_index[c["title"].lower()].append(c)
        self.corpus_lower = "\n".join(c["text"] for c in self.chunks).lower()

    # --- тяжёлые модели грузим только когда они нужны
    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            device = self.rerank_device
            if device is None:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
        return self._reranker

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        return self._bm25

    # --- поиск
    def embed(self, query: str):
        """Эмбеддинг вопроса (bge-m3 через Ollama), нормированный."""
        q = np.array(self.client.embed(model=EMBED_MODEL, input=query)["embeddings"][0], dtype="float32")
        return q / np.linalg.norm(q)

    def dense_top(self, query: str, n: int):
        """n ближайших чанков из Chroma: [(косинусная близость, id чанка), ...] от лучшего к худшему."""
        r = self.col.query(query_embeddings=[self.embed(query).tolist()], n_results=n, include=["distances"])
        return [(1.0 - d, int(i)) for i, d in zip(r["ids"][0], r["distances"][0])]

    def search(self, query: str, k: int = 5, mode: str = "dense"):
        """Возвращает список пар (оценка, чанк). mode: dense | hybrid (dense + BM25, RRF).

        Оценка в режиме dense это косинусная близость, в режиме hybrid это сумма RRF.
        """
        if mode == "hybrid":
            fused = defaultdict(float)
            for rank, (_s, i) in enumerate(self.dense_top(query, 50)):
                fused[i] += 1 / (60 + rank + 1)
            b = self.bm25.get_scores(tok(query))
            for rank, i in enumerate(np.argsort(b)[::-1][:50]):
                fused[int(i)] += 1 / (60 + rank + 1)
            top = sorted(fused, key=fused.get, reverse=True)[:k]
            return [(fused[i], self.chunks[i]) for i in top]
        return [(s, self.chunks[i]) for s, i in self.dense_top(query, k)]

    # --- сущности: имена вида torch.xxx.yyy из вопроса
    def unknown_names(self, query: str) -> list[str]:
        names = re.findall(r"torch(?:\.\w+)+", query)
        return [n for n in names
                if n.lower() not in self.title_index and n.lower() not in self.corpus_lower]

    def named_chunks(self, query: str, per_name: int = 2):
        out = []
        for n in re.findall(r"torch(?:\.\w+)+", query):
            out += self.title_index.get(n.lower(), [])[:per_name]
        return out

    def retrieve(self, query: str, k: int = 5, mode: str = "dense", gate_pool: int = GATE_POOL,
                 graph: bool = False):
        """Возвращает (причина отказа или None, список пар (оценка, чанк)).

        1) в вопросе есть имя torch.*, которого нет в документации -> отказ;
        2) чанки с точным именем из вопроса закрепляются первыми (оценка 1.0) и порог не проверяется;
        3) иначе реранкер оценивает gate_pool dense-кандидатов; лучшая оценка ниже порога -> отказ;
        4) контекст добирается из dense до k фрагментов (порядок dense, реранкер на него не влияет);
        5) graph=True: к концу контекста добавляются соседи по ссылкам из docstring'ов (оценка 0.0).
        """
        bad = self.unknown_names(query)
        if bad:
            return f"в документации нет имени {', '.join(bad)}", []
        pinned = self.named_chunks(query)
        wide = self.search(query, max(gate_pool, k), mode)
        if not pinned:
            scores = self.reranker.predict([(query, c["text"]) for _s, c in wide])
            if max(scores) < self.thresh:
                return "в документации не нашлось подходящего раздела", []
        res = [(1.0, c) for c in pinned][:k]
        ids = {c["id"] for _s, c in res}
        for s, c in wide:
            if len(res) >= k:
                break
            if c["id"] in ids:
                continue
            res.append((s, c))
            ids.add(c["id"])
        if graph:
            res = self.expand(res)
        return None, res

    # --- граф связей
    @property
    def graph(self):
        """(исходящие рёбра, число входящих рёбер, узлы) из graph.json; грузится при первом обращении."""
        if self._graph is None:
            if not GRAPH_PATH.exists():
                raise FileNotFoundError("нет graph.json: запусти  python rag.py graph")
            with open(GRAPH_PATH, encoding="utf-8") as f:
                g = json.load(f)
            out = defaultdict(list)
            indeg = defaultdict(int)
            for a, b in g["edges"]:
                out[a].append(b)
                indeg[b] += 1
            self._graph = (out, indeg, g["nodes"])
        return self._graph

    def expand(self, res, top: int = GRAPH_TOP, per_chunk: int = GRAPH_PER_CHUNK,
               max_indeg: int = GRAPH_MAX_INDEG):
        """Добавляет в конец контекста соседей лучших чанков по ссылкам из docstring'ов (оценка 0.0).

        Хабы (узлы, на которые ссылаются max_indeg раз и чаще, вроде torch.Tensor) не подтягиваются.
        """
        out, indeg, nodes = self.graph
        ids = {c["id"] for _s, c in res}
        extra = []
        for _s, c in res[:top]:
            added = 0
            for dst in out.get(node_name(c), []):
                if added >= per_chunk:
                    break
                if indeg[dst] >= max_indeg:
                    continue
                for cid in nodes.get(dst, []):
                    if cid not in ids:
                        extra.append((0.0, self.chunks[cid]))
                        ids.add(cid)
                        added += 1
                        break
        return res + extra

    # --- генерация
    def build_messages(self, query: str, res, max_chars: int = 3000):
        user = (f"Excerpts:\n{build_context(res, max_chars)}\n\n"
                f"Question: {query}\n\nAnswer in {answer_language(query)}.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def generate(self, query: str, res, model: str | None = None):
        """Генератор: отдаёт ответ модели по кускам (для стриминга)."""
        stream = self.client.chat(model=model or self.gen_model, messages=self.build_messages(query, res),
                                  stream=True, options={"temperature": 0.1, "num_ctx": 8192})
        for part in stream:
            yield part["message"]["content"]

    def answer(self, query: str, k: int = 5, mode: str = "dense", model: str | None = None, echo: bool = True,
               graph: bool = False):
        """Полный цикл. Возвращает (текст ответа, [(заголовок, файл), ...])."""
        reason, res = self.retrieve(query, k, mode, graph=graph)
        if reason:
            msg = refusal_message(query, reason)
            if echo:
                print(msg)
            return msg, []
        text = ""
        for piece in self.generate(query, res, model):
            if echo:
                print(piece, end="", flush=True)
            text += piece
        if echo:
            print()
        return text, [(c["title"], c["file"]) for _s, c in res]


# ----------------------------------------------------------------------------- проверка

def run_eval(rag: RAG, k: int = 5):
    """Проверка поиска и отказа: попал ли нужный чанк в контекст и отклонены ли вопросы без ответа."""
    for name in ("eval.json", "eval_holdout.json"):
        path = DATA / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        n_gold = sum(1 for it in items if it["gold"])
        n_none = len(items) - n_gold
        in_ctx = refused = 0
        sizes, problems = [], []
        for it in items:
            reason, res = rag.retrieve(it["q"], k)
            if it["gold"]:
                if reason:
                    problems.append(("ЛИШНИЙ ОТКАЗ", it["q"], reason))
                    continue
                sizes.append(len(res))
                if any(is_hit(c["title"], it["gold"]) for _s, c in res):
                    in_ctx += 1
                else:
                    problems.append(("НЕ НАШЁЛ", it["q"], [c["title"][:50] for _s, c in res[:3]]))
            elif reason:
                refused += 1
            else:
                problems.append(("НЕ ОТКАЗАЛ", it["q"], res[0][1]["title"][:60]))
        avg = round(sum(sizes) / len(sizes), 1) if sizes else 0
        print(f"== {name}: вопросов {len(items)}")
        print(f"   gold в контексте: {in_ctx} из {n_gold} | фрагментов в среднем: {avg}")
        print(f"   вопросов без ответа отклонено: {refused} из {n_none}")
        for p in problems:
            print("  ", p)


def run_graph_eval(rag: RAG):
    """eval_graph.json: вопросы, ответ на которые лежит в связанных чанках. Покрытие gold без графа и с ним."""
    path = DATA / "eval_graph.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    base = grafted = 0.0
    n = helped = refused = 0
    for it in items:
        reason, res = rag.retrieve(it["q"])
        if reason:
            refused += 1
            continue
        res2 = rag.expand(res)
        b, a = covered(it, res), covered(it, res2)
        base += b
        grafted += a
        n += 1
        helped += a > b
    print(f"== eval_graph.json: вопросов {len(items)} | отказов {refused}")
    if n:
        print(f"   покрытие gold без графа: {base / n:.2f} | с графом: {grafted / n:.2f} | граф помог в {helped} вопросах")


# ----------------------------------------------------------------------------- консоль

def main():
    import argparse
    p = argparse.ArgumentParser(description="RAG-ассистент по документации PyTorch")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("ask", help="задать вопрос")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    a.add_argument("--model", default=None)
    a.add_argument("--graph", action="store_true", help="расширять контекст по графу связей")
    sub.add_parser("index", help="заполнить Chroma (сервер в Docker) из chunks_full.json и emb.npy")
    sub.add_parser("eval", help="проверить поиск и отказ на eval.json и eval_holdout.json, граф на eval_graph.json")
    sub.add_parser("graph", help="пересобрать graph.json из docs.json и chunks_full.json")
    args = p.parse_args()

    if args.cmd == "ask":
        rag = RAG()
        text, sources = rag.answer(args.question, k=args.k, mode=args.mode, model=args.model, graph=args.graph)
        for i, (title, file) in enumerate(sources, 1):
            print(f"[{i}] {title}  ({file})")
        if sources:
            cited, invented = check_citations(text, len(sources))
            print("ссылки:", cited, "| выдуманные номера:", invented)
    elif args.cmd == "index":
        import time
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(EMB_PATH)
        if emb.shape[0] != len(chunks):
            raise SystemExit(f"emb.npy ({emb.shape[0]}) не соответствует chunks_full.json ({len(chunks)}): "
                             "пересчитай эмбеддинги после пересборки корпуса (02_search.ipynb)")
        if any(c["id"] != i for i, c in enumerate(chunks)):
            raise SystemExit("id чанков должны совпадать с их номерами в списке")
        t0 = time.time()
        col = build_index(chunks, emb)
        print(f"в базу записано {col.count()} векторов за {time.time() - t0:.0f} с -> Chroma {CHROMA_HOST}:{CHROMA_PORT}")
        # самопроверка: вектор чанка, поданный как запрос, должен находить сам этот чанк (расстояние ~0)
        sample = np.random.default_rng(0).choice(len(chunks), 20, replace=False)
        ok = sum(col.query(query_embeddings=[emb[i].tolist()], n_results=1, include=["distances"])["distances"][0][0] < 1e-3
                 for i in sample)
        print(f"самопроверка: {ok} из 20")
    elif args.cmd == "eval":
        rag = RAG()
        run_eval(rag)
        if GRAPH_PATH.exists():
            run_graph_eval(rag)
    elif args.cmd == "graph":
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        with open(DOCS_PATH, encoding="utf-8") as f:
            docs = json.load(f)
        graph, stats = build_graph(chunks, docs)
        with open(GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False)
        for role, (total, ok) in stats.items():
            print(f"{role}: ссылок {total}, разрешено {ok} ({ok / total:.0%})")
        print(f"рёбер: {len(graph['edges'])} | узлов: {len(graph['nodes'])} -> {GRAPH_PATH.name}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
