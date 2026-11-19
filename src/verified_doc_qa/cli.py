"""Command line:  docqa ingest PATH...  |  docqa ask "question"  |  docqa docs  |  docqa eval"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docqa", description="Verified Doc QA")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index files or folders (.md, .txt, .pdf)")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--reset", action="store_true", help="drop the collection first")
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--json", action="store_true")
    sub.add_parser("docs", help="list indexed documents")
    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    if args.cmd == "eval":
        from .config import ROOT

        sys.path.insert(0, str(ROOT))  # evals/ lives next to src/, not inside the package
        from evals.run_eval import run

        run(limit=args.limit)
        return

    from .config import get_settings
    from .graph import build_store

    settings = get_settings()
    store = build_store(settings)
    if args.cmd == "ingest":
        from .ingest import index_paths
        from .llm import get_chat_model

        if args.reset:
            store.reset()
        llm = get_chat_model(settings) if settings.contextual else None
        for source, n in index_paths(store, args.paths, settings.chunk_size, settings.chunk_overlap, llm).items():
            print(f"{n:4d} chunks  {source}")
    elif args.cmd == "docs":
        for source, n in store.sources().items():
            print(f"{n:4d} chunks  {source}")
    else:
        from .graph import DocQA

        resp = DocQA(settings, store=store).ask(" ".join(args.question))
        if args.json:
            print(json.dumps(resp.model_dump(), indent=2, default=str))
            return
        badge = {"answered": "verified", "unverified": "NOT verified", "refused": "refused"}[resp.status]
        print(f"{resp.answer}\n")
        for c in resp.citations:
            print(f"[{c.n}] {c.source}{f' - {c.section}' if c.section else ''}")
        print(f"\n[{badge} · relevance {resp.relevance} · {resp.attempts} attempt(s) · {resp.seconds}s]")


if __name__ == "__main__":
    main()
