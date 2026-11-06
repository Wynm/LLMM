"""واجهة سطر الأوامر لمِشكاة.

    python -m mishkat.cli build
    python -m mishkat.cli search "كيف قاس البيروني الأرض؟"
    python -m mishkat.cli stats
    python -m mishkat.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import CorpusError, load_corpus
from .engine import ARTIFACT_DIR, Engine


# ------------------------------------------------------------------ مساعدات
def _c(text: str, code: str) -> str:
    """تلوين الطرفية إن كانت تدعمه."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD, DIM, CYAN, GREEN, YELLOW, MAGENTA = "1", "2", "36", "32", "33", "35"


def _hr(char: str = "─", width: int = 68) -> str:
    return _c(char * width, DIM)


def _print_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> None:
    widths = [
        max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(_hr("─", sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ------------------------------------------------------------------ الأوامر
def cmd_build(args: argparse.Namespace) -> int:
    print(_c("▸ بناء مِشكاة", BOLD))
    t0 = time.perf_counter()
    corpus = load_corpus(args.corpus)
    print(f"  المجموعة: {len(corpus)} مستندًا، {len(corpus.entities)} كيانًا")
    eng = Engine.build(
        corpus,
        dim=args.dim,
        window=args.window,
        verbose=True,
    )
    eng.save(args.out)
    total = time.perf_counter() - t0
    print(_c(f"✔ تمّ البناء في {total:.2f} ثانية → {args.out}", GREEN))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    eng = Engine.load_or_build(args.out)
    hits, trace = eng.search(args.query, k=args.k, mode=args.mode, domain=args.domain, with_trace=True)
    t = trace.to_dict()
    print(_hr())
    print(f"{_c('سؤال:', BOLD)} {args.query}")
    print(
        f"{_c('تحليل:', DIM)} سيقان={t['terms']} جذور={t['roots'][:6]}"
        f" | زمن: معجمي {t['timing_ms']['lexical']}ms"
        f" دلالي {t['timing_ms']['semantic']}ms"
        f" دمج {t['timing_ms']['fuse']}ms"
    )
    print(_hr())
    if not hits:
        print("لا نتائج.")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"{i}. {_c(h.title, BOLD)}  {_c('[' + h.domain + ']', MAGENTA)}")
        print(
            f"   {_c('درجة', DIM)} {h.score:.4f}  "
            f"{_c('معجمي', DIM)} {h.lexical:.2f}  {_c('دلالي', DIM)} {h.semantic:.2f}  "
            f"{_c(h.doc_id, DIM)}"
        )
        if h.snippet:
            print(f"   {h.snippet[:160]}")
        print()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    eng = Engine.load_or_build(args.out)
    st = eng.stats()
    c = st["corpus"]
    print(_c("▸ مِشكاة — إحصاءات", BOLD))
    print(_hr())
    _print_table(
        [
            ("المستندات", str(c["documents"])),
            ("الكيانات", str(c["entities"])),
            ("العلاقات", str(c["edges"])),
            ("الرموز (tokens)", str(c["tokens"])),
            ("الأنماط (types)", str(c["types"])),
            ("متوسط طول المستند", str(c["avg_doc_tokens"])),
        ],
        ("المقياس", "القيمة"),
    )
    print()
    print(_c("النطاقات:", BOLD))
    for name, count in c["domains"].items():
        bar = "▇" * min(30, count)
        print(f"  {name:<14s} {bar} {count}")
    print()
    print(_c("الفهرس:", BOLD))
    for k, v in st["index"].items():
        print(f"  {k:<18s} {v}")
    print()
    print(_c("الفضاء الدلالي:", BOLD), st["space"])
    print(_c("الرسم:", BOLD), {k: v for k, v in st["graph"].items() if k != "relations"})
    print(_c("أبرز العلاقات:", BOLD))
    for rel, n in list(st["graph"]["relations"].items())[:10]:
        print(f"  {rel:<16s} {n}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    eng = Engine.load_or_build(args.out)
    result = eng.graph_query(args.query)
    print(_hr())
    print(f"{_c('استعلام:', BOLD)} {args.query}")
    timing = f"({result['elapsed_ms']}ms)"
    print(f"{_c('نتائج:', BOLD)} {result['count']}  {_c(timing, DIM)}")
    print(_hr())
    for b in result["bindings"][: args.limit]:
        pretty = [f"{k}={eng.graph.nodes[v].label if v in eng.graph.nodes else v}" for k, v in b.items()]
        print("  " + "  |  ".join(pretty))
    if not result["bindings"]:
        print("لا نتائج.")
    return 0


def cmd_roots(args: argparse.Namespace) -> int:
    from .text.analyze import analyze
    from .text.roots import root_candidates

    for word in args.words:
        a = analyze(word, with_alternatives=True)
        alts = ", ".join(a.alternatives) if a.alternatives else "—"
        print(f"{_c(word, BOLD):<20s} جذر={_c(a.root, GREEN):<8s} ساق={a.stem:<12s} نوع={a.kind:<6s} ثقة={a.confidence:.2f}")
        print(f"{'':<20s} بدائل: {alts}  |  مرشّحات: " + ", ".join(
            f"{c.root}({c.score:.1f})" for c in root_candidates(word, limit=4)
        ))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """تقرير أداء: زمن كل مرحلة + استدعاء HNSW مقابل البحث الشامل."""
    from .bench import run_benchmark

    report = run_benchmark(queries=args.queries)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_corpus_check(args: argparse.Namespace) -> int:
    try:
        corpus = load_corpus(args.corpus)
    except CorpusError as exc:
        print(_c(f"✗ خطأ في المجموعة: {exc}", YELLOW))
        return 2
    st = corpus.stats()
    print(_c("✔ المجموعة سليمة", GREEN))
    print(f"  مستندات: {st['documents']} | كيانات: {st['entities']} | علاقات: {st['edges']}")
    print(f"  نطاقات: {len(st['domains'])} | رموز: {st['tokens']}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """تقييم جودة الاسترجاع على مجموعةٍ مُعنونة — أرقامٌ لا انطباعات."""
    from .evaluate import load_queries, run_eval
    from .engine import Engine

    engine = Engine.load(args.artifacts)
    queries = load_queries(args.queries) if args.queries else None
    qs = queries or load_queries()
    print(_c("▸ تقييم الاسترجاع", BOLD))
    print(f"  {len(qs)} استعلامًا مُعنونًا · k = {args.k}")
    reports = run_eval(engine, queries, k=args.k)
    print()
    for rep in reports:
        print("  " + rep.row())
        for line in rep.group_rows():
            print(line)
    best = max(reports, key=lambda r: (r.mrr, r.recall))
    print()
    print(_c(f"  الأفضل في MRR: {best.strategy} ({best.mrr:.3f})", GREEN))
    shipped = next((r for r in reports if r.strategy == "هجين (المُشحون)"), None)
    if args.misses and shipped and shipped.misses:
        print()
        print(_c(f"  إخفاقاتٌ كاملة ({len(shipped.misses)} من {shipped.n}):", BOLD))
        for m in shipped.misses[: args.misses]:
            got = ", ".join(m["got"][:3]) or "لا شيء"
            print(f"   ✗ {m['query']}  ← توقّعنا {', '.join(m['expect'])}، جاء {got}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """تدريب المحوّل العصبي على نصوص المجموعة."""
    from .lm.model import Config
    from .lm.sample import MODEL_DIR
    from .lm.train import TrainConfig, train

    corpus = load_corpus(args.corpus)
    text = "\n".join(d.title + ". " + d.summary + " " + d.body for d in corpus.documents)
    print(_c("▸ تدريب المحوّل الحرفي", BOLD))
    print(f"  نصّ: {len(text)} حرفًا · {len(corpus)} مستندًا")
    cfg = Config(vocab_size=200, block_size=args.block, n_layer=args.layers, n_head=4, n_embd=args.dim)

    def progress(row: dict) -> None:
        print(
            f"  خطوة {row['step']:>5d} | خسارة {row['loss']:.3f} | حجز {row['val_loss']:.3f}"
            f" | lr {row['lr']:.5f} | {row['elapsed_s']:.1f}s"
        )

    report = train(
        text,
        MODEL_DIR,
        cfg=cfg,
        tcfg=TrainConfig(steps=args.steps, batch_size=args.batch, lr=args.lr),
        progress=progress,
    )
    loss = f"{report['final_loss']:.3f}"
    val = f"{report['final_val_loss']:.3f}"
    print(_c(f"✔ اكتمل التدريب: خسارة {loss} · حجز {val}", GREEN))
    print(f"  {report['params']:,} وسيطًا · {report['steps_per_second']} خطوة/ثانية · {report['elapsed_s']}s")
    print(f"  حُفظ في {args.out} — جرّب: python -m mishkat.cli generate \"العلم\"")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .lm.sample import generate

    print(_c("▸ توليد من المحوّل", BOLD))
    text = generate(
        args.prompt,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(_hr())
    print(text)
    print(_hr())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(_c(f"▸ مِشكاة تعمل على http://0.0.0.0:{args.port}", GREEN))
    print(_c("  (الاستوديو على الجذر، وواجهة API على /api)", DIM))
    # بناء المحرّك مسبقًا حتى يكون أول طلبٍ سريعًا
    Engine.load_or_build(args.out)
    uvicorn.run(
        "mishkat.api.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .text.analyze import analyze_text
    from .text.tokenize import tokenize

    text = args.text or sys.stdin.read()
    rows = analyze_text(text)
    print(_hr())
    for tok, a in rows:
        flags = "•" if a.is_stop else " "
        print(
            f"{flags} {tok.raw:<14s} → {a.norm:<14s} ساق={a.stem:<12s} جذر={_c(a.root, GREEN):<8s} {a.kind}"
        )
    print(_hr())
    print(f"كلمات: {len(rows)} | جمل: {len(tokenize(text)) and len(set(t.sentence for t in tokenize(text)))}")
    return 0


# ------------------------------------------------------------------ التحليل
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mishkat", description="مِشكاة — محرّك معرفةٍ عربيٍّ محلّي")
    p.add_argument("--out", default=str(ARTIFACT_DIR), help="مجلّد المخرجات")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="بناء كل الفهارس")
    b.add_argument("--corpus", default=None)
    b.add_argument("--dim", type=int, default=96)
    b.add_argument("--window", type=int, default=5)
    b.add_argument("--all", action="store_true", help="(متوافق مع make build)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="بحث")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "semantic"])
    s.add_argument("--domain", default=None)
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("stats", help="إحصاءات")
    st.set_defaults(func=cmd_stats)

    g = sub.add_parser("graph", help="استعلام الرسم")
    g.add_argument("query")
    g.add_argument("--limit", type=int, default=20)
    g.set_defaults(func=cmd_graph)

    r = sub.add_parser("roots", help="تحليل جذور كلمات")
    r.add_argument("words", nargs="+")
    r.set_defaults(func=cmd_roots)

    a = sub.add_parser("analyze", help="تحليل نصّ")
    a.add_argument("text", nargs="?")
    a.set_defaults(func=cmd_analyze)

    bb = sub.add_parser("bench", help="تقرير أداء")
    bb.add_argument("--queries", type=int, default=40)
    bb.set_defaults(func=cmd_bench)

    ev = sub.add_parser("eval", help="تقييم جودة الاسترجاع")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--queries", default=None, help="ملفّ استعلامات بديل (JSONL)")
    ev.add_argument("--artifacts", default="artifacts", help="مجلّد المخرجات")
    ev.add_argument("--misses", type=int, default=0, help="اعرض أول n إخفاقٍ كامل")
    ev.set_defaults(func=cmd_eval)

    cc = sub.add_parser("corpus-check", help="تحقّق من المجموعة")
    cc.add_argument("--corpus", default=None)
    cc.set_defaults(func=cmd_corpus_check)

    tr = sub.add_parser("train", help="تدريب المحوّل العصبي")
    tr.add_argument("--steps", type=int, default=1500)
    tr.add_argument("--batch", type=int, default=12)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--block", type=int, default=64)
    tr.add_argument("--dim", type=int, default=96)
    tr.add_argument("--layers", type=int, default=2)
    tr.add_argument("--corpus", default=None)
    tr.set_defaults(func=cmd_train)

    ge = sub.add_parser("generate", help="توليد نصّ")
    ge.add_argument("prompt", nargs="?", default="العلم")
    ge.add_argument("--length", type=int, default=200)
    ge.add_argument("--temperature", type=float, default=0.9)
    ge.add_argument("--top-k", dest="top_k", type=int, default=12)
    ge.set_defaults(func=cmd_generate)

    sv = sub.add_parser("serve", help="تشغيل الاستوديو")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="0.0.0.0")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "build" and getattr(args, "all", False):
        pass
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
