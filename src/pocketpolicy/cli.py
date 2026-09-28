"""pocketpolicy command line.

    pocketpolicy campaign --budget 24 --planner committee
    pocketpolicy emit --width 64 --depth 3 --H 4 --dagger 4 -o out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loopgraph import (Campaign, Committee, GridDecider, Ledger, Objective, ParetoDecider,
                       RandomDecider, at_most, equals)

from .pipeline import DEFAULTS, SPACE, execute, graph

OBJECTIVES = [Objective("success"), Objective("flash_kb", maximize=False)]
GATES = [
    equals("bit_exact", 1, "emitted C must match the integer reference byte for byte"),
    at_most("m4_duty_est", 0.5, "inference may use at most half the MCU at the control rate"),
]
GOAL = ("Distil a pick-and-place teacher for the SO-101 arm into an int8 MLP that runs on "
        "a microcontroller. width/depth set network size, H is the action-chunk length "
        "(half of each chunk runs before re-planning), dagger is the number of DAgger "
        "rounds. Find the smallest flash footprint at each level of closed-loop success.")


def planner(name: str, seed: int):
    pareto = ParetoDecider(OBJECTIVES, seed=seed)
    if name == "grid":
        return GridDecider()
    if name == "random":
        return RandomDecider(seed)
    if name == "pareto":
        return pareto
    if name in ("llm", "committee"):
        from loopgraph.llm import LLMDecider
        members = [pareto, RandomDecider(seed)]
        if LLMDecider.available():
            llm = LLMDecider(OBJECTIVES, goal=GOAL, fallback=pareto)
            if name == "llm":
                return llm
            members.insert(0, llm)
        elif name == "llm":
            print("LLM planner unavailable (no anthropic SDK or API key); using pareto",
                  file=sys.stderr)
            return pareto
        return Committee(members)
    raise SystemExit(f"unknown planner {name}")


def cmd_campaign(a) -> int:
    L = Ledger(a.ledger)

    def log(e):
        m = e.metrics
        print(f"[{e.decided_by:>14}] {e.params} -> success {m['success']:.3f} "
              f"(float {m['success_float']:.3f}) {m['flash_kb']:.1f} KB "
              f"{'ok' if e.admitted else 'REJECTED'}", flush=True)

    seed_runs = [{"width": 256, "depth": 3, "H": 4, "dagger": 4},
                 {"width": 16, "depth": 2, "H": 1, "dagger": 0}]
    if not L.entries(a.experiment):  # anchor the frontier at both ends
        from loopgraph.ledger import record
        for p in seed_runs:
            m, k = execute(p, a.seed, a.cache)
            log(record(L, a.experiment, p, m, GATES, keys=k, decided_by="seed",
                       rationale="frontier anchor"))
    Campaign(a.experiment, SPACE, lambda p: execute(p, a.seed, a.cache), L,
             planner(a.planner, a.seed), GATES, budget=a.budget, batch=a.batch,
             on_entry=log).run()
    return 0


def cmd_ablate(a) -> int:
    """Controlled DAgger ablation: same architectures, rounds 0 vs N, several seeds."""
    from loopgraph.ledger import record

    L = Ledger(a.ledger)
    done = {(tuple(sorted(e.params.items()))) for e in L.entries("dagger_ablation")}
    for width, depth in ((32, 2), (64, 2), (64, 3)):
        for dagger in (0, 4):
            for seed in range(a.seeds):
                p = {"width": width, "depth": depth, "H": 1, "dagger": dagger, "seed": seed}
                if tuple(sorted(p.items())) in done:
                    continue
                m, k = execute({x: v for x, v in p.items() if x != "seed"}, seed, a.cache)
                record(L, "dagger_ablation", p, m, GATES, keys=k, decided_by="design",
                       rationale="fixed ablation grid")
                print(p, f"success {m['success']:.3f}", flush=True)
    return 0


def cmd_emit(a) -> int:
    p = {**DEFAULTS, "width": a.width, "depth": a.depth, "H": a.H, "dagger": a.dagger,
         "seed": a.seed}
    r = graph(a.cache).run(["deployed", "metrics"], p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn, text in r["deployed"].emit(a.name).items():
        (out / fn).write_text(text)
    m = r["metrics"]
    print(f"wrote {out}/: success {m['success']:.3f}, {m['flash_kb']:.1f} KB flash, "
          f"{m['ram_b']} B RAM, bit_exact={m.get('bit_exact')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pocketpolicy")
    p.add_argument("--cache", default=".loopgraph/cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("campaign")
    s.add_argument("--ledger", default="results/ledger.jsonl")
    s.add_argument("--experiment", default="distill")
    s.add_argument("--planner", default="committee",
                   choices=["committee", "pareto", "grid", "random", "llm"])
    s.add_argument("--budget", type=int, default=24)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_campaign)

    s = sub.add_parser("ablate")
    s.add_argument("--ledger", default="results/ledger.jsonl")
    s.add_argument("--seeds", type=int, default=3)
    s.set_defaults(fn=cmd_ablate)

    s = sub.add_parser("emit")
    for k, d in (("width", 64), ("depth", 3), ("H", 4), ("dagger", 4), ("seed", 0)):
        s.add_argument(f"--{k}", type=int, default=d)
    s.add_argument("--name", default="pocket")
    s.add_argument("-o", "--out", default="out")
    s.set_defaults(fn=cmd_emit)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
