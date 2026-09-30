"""Sim-to-real stress test of the deployed int8 policies.

    pocketpolicy sim2real --seeds 2 -o results/sim2real.json --readme README.md

Two configurations from the frontier, each trained two ways (the published
recipe, and domain-randomised DAgger), every one run as its integer pipeline
(`DeployedPolicy`) under the error presets of `realworld`. Students come from
the same content-addressed cache as the campaign, so the published ones are
not retrained. The README section is rendered from the JSON, never by hand.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from loopgraph import Graph, node

from .deploy import DeployedPolicy
from .pipeline import DEFAULTS, EVAL_SEED, _env, calib, exec_k, student
from .realworld import DEG, PRESETS, Perturb, describe, rollout_real
from .student import distill

CONFIGS = {  # name: (width, depth, H, dagger), both on the published frontier
    "6.4 KB": (64, 2, 1, 2),
    "144 KB": (256, 3, 4, 2),
}
TRAININGS = {  # name: teacher view for domain-randomised DAgger (None: published recipe)
    "published": None,
    "robust, teacher sees truth": "now",
    "robust, teacher sees truth at reading": "read",
    "robust, teacher sees reading": "observed",
}
ROBUST_PRESET = "realistic"   # training draws each episode's errors up to 1.25x this
CONDITIONS = {  # name: (perturbation, command-relative goals)
    "ideal": (PRESETS["ideal"], False),
    "ideal + fix": (PRESETS["ideal"], True),
    "sag 1°, no fix": (Perturb(sag=1 * DEG), False),
    "sag 1° + fix": (Perturb(sag=1 * DEG), True),
    "realistic, no fix": (PRESETS["realistic"], False),
    "realistic + fix": (PRESETS["realistic"], True),
    "pessimistic + fix": (PRESETS["pessimistic"], True),
}
BEGIN, END = "<!-- sim2real:begin -->", "<!-- sim2real:end -->"


@node(version="1")
def robust_student(width, depth, H, dagger, seed, train_episodes, epochs, robust,
                   teacher_view):
    env, T = _env()
    net, info = distill(env, T, width, depth, H, dagger, episodes=train_episodes,
                        epochs=epochs, seed=seed, robust=PRESETS[robust],
                        teacher_view=teacher_view)
    return {"net": net, "info": info}


def graph(cache_dir=".loopgraph/cache") -> Graph:
    return Graph([student, robust_student, calib], cache_dir=cache_dir)


def evaluate(policy, H: int, episodes: int) -> dict[str, float]:
    env, _ = _env()
    return {c: float(rollout_real(env, policy, episodes, EVAL_SEED, exec_k(H), p,
                                  rel).success.mean())
            for c, (p, rel) in CONDITIONS.items()}


def cell(config: str, training: str, seed: int, episodes: int, cache_dir, sizes=None) -> dict:
    """Train (or fetch from cache), deploy as int8, evaluate under every condition."""
    width, depth, H, dagger = CONFIGS[config]
    p = {**DEFAULTS, **(sizes or {}), "width": width, "depth": depth, "H": H,
         "dagger": dagger, "seed": seed, "robust": ROBUST_PRESET,
         "teacher_view": TRAININGS[training]}
    target = "student" if TRAININGS[training] is None else "robust_student"
    r = graph(cache_dir).run([target, "calib"], p)
    d = DeployedPolicy.from_student(r[target]["net"], r["calib"])
    return {"config": config, "training": training, "seed": seed,
            "flash_kb": round(d.flash_bytes / 1024, 2),
            "success": evaluate(d.policy, H, episodes)}


def _cell(args):
    return cell(*args)


def run(seeds: int = 2, episodes: int = 400, cache_dir=".loopgraph/cache", jobs: int = 1,
        sizes=None, log=print) -> dict:
    env, T = _env()
    teacher = evaluate(lambda s: T.chunk(s, 1), 1, episodes)
    log(f"teacher {teacher}")
    work = [(c, t, s, episodes, cache_dir, sizes) for c in reversed(CONFIGS) for t in TRAININGS
            for s in range(seeds)]  # biggest first, so parallel workers finish together
    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            for row in ex.map(_cell, work):
                log(row)
                rows.append(row)
    else:
        for w in work:
            rows.append(_cell(w))
            log(rows[-1])
    return {"eval_episodes": episodes, "eval_seed": EVAL_SEED, "seeds": seeds,
            "robust_preset": ROBUST_PRESET, "robust_scale": 1.25,
            "presets": {k: describe(v) for k, v in PRESETS.items()},
            "conditions": list(CONDITIONS), "teacher": teacher, "rows": rows}


# -- report ------------------------------------------------------------------
def means(res: dict) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for c in CONFIGS:
        for t in TRAININGS:
            rs = [r["success"] for r in res["rows"] if r["config"] == c and r["training"] == t]
            if rs:
                out[(c, t)] = {k: float(np.mean([r[k] for r in rs])) for k in res["conditions"]}
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def render(res: dict) -> str:
    m = means(res)
    conds = res["conditions"]
    n = res["seeds"]
    rp = res["presets"]["realistic"]
    pp = res["presets"]["pessimistic"]
    lines = [
        BEGIN,
        "## Sim-to-real: what survives",
        "",
        "The simulator hands the policy perfect numbers. A real arm will not. We fed "
        "the policy corrupted readings while the physics ran on the true state, "
        f"{res['eval_episodes']} held-out episodes per cell, int8 pipeline, "
        f"mean of {n} training seed{'s' if n > 1 else ''} "
        "(`pocketpolicy sim2real`, `results/sim2real.json`).",
        "",
        "| policy | training | " + " | ".join(conds) + " |",
        "|---|---|" + "---|" * len(conds),
        "| teacher | scripted | " + " | ".join(_pct(res["teacher"][c]) for c in conds) + " |",
    ]
    for (c, t), row in m.items():
        lines.append(f"| {c} | {t} | " + " | ".join(_pct(row[k]) for k in conds) + " |")
    small, big = m.get(("6.4 KB", "published")), m.get(("144 KB", "published"))
    lines += [
        "",
        f"- **realistic**: detector noise {rp['det_noise'] * 1e3:.0f} mm and bias "
        f"{rp['det_bias'] * 1e3:.0f} mm, {rp['latency']:.0f} ticks of latency, "
        f"{rp['joint_cal'] / DEG:.1f}° joint calibration error, {rp['sag'] / DEG:.0f}° gravity "
        f"sag, a {rp['grasp_radius'] * 1e2:.1f} cm grasp window, grasp flag wrong "
        f"{rp['flag_err'] * 100:.0f}% of ticks. **pessimistic**: {pp['det_noise'] * 1e3:.0f} mm, "
        f"{pp['det_bias'] * 1e3:.0f} mm, {pp['latency']:.0f} ticks, {pp['joint_cal'] / DEG:.0f}°, "
        f"{pp['sag'] / DEG:.0f}°, {pp['grasp_radius'] * 1e2:.1f} cm, "
        f"{pp['flag_err'] * 100:.0f}%. These magnitudes are guesses, not measurements.",
    ]
    if small and big:
        lines += [
            f"- **Gravity sag broke the original control loop.** The policy sent "
            "`measured position + step`. A servo holding weight stops a little short of "
            "its target, so the arm stalled short of every object: with 1° of sag, "
            f"success fell to {_pct(small['sag 1°, no fix'])} (6.4 KB), "
            f"{_pct(big['sag 1°, no fix'])} (144 KB) and "
            f"{_pct(res['teacher']['sag 1°, no fix'])} for the teacher. "
            "The fix: add the step to the last command sent, not to the "
            f"reading. That brings it back to {_pct(small['sag 1° + fix'])} and "
            f"{_pct(big['sag 1° + fix'])}. `firmware/so101_main.c` now does this "
            "(`firmware/cmd_relative.c`), and a test checks that the C and Python "
            "versions produce the same numbers.",
            "- **The big model holds up "
            + ("better" if big["realistic + fix"] > small["realistic + fix"] else "no better")
            + ".** With realistic errors and the fix: "
            f"{_pct(big['realistic + fix'])} for 144 KB, {_pct(small['realistic + fix'])} "
            f"for 6.4 KB. With pessimistic errors: {_pct(big['pessimistic + fix'])} and "
            f"{_pct(small['pessimistic + fix'])}.",
        ]
    if small and big:
        lines.append(
            "- **Training on noisy readings (domain-randomised DAgger).** Each training "
            f"episode got its own random errors, from none up to {res['robust_scale']:.2f}x "
            "realistic, and ran with the fix. The student learned from what it saw. We tried "
            "three choices of what the teacher sees when it labels. Points gained over the "
            "published model (6.4 KB / 144 KB):")
        for t in TRAININGS:
            if t == "published" or ("6.4 KB", t) not in m or ("144 KB", t) not in m:
                continue
            a, b = m[("6.4 KB", t)], m[("144 KB", t)]
            d = lambda k: f"{100 * (a[k] - small[k]):+.0f} / {100 * (b[k] - big[k]):+.0f}"
            lines.append(f"  - {t}: ideal {d('ideal')}, realistic + fix "
                         f"{d('realistic + fix')}, pessimistic + fix {d('pessimistic + fix')}.")
        gains = [100 * (m[(c, t)]["realistic + fix"] - m[(c, "published")]["realistic + fix"])
                 for c in CONFIGS for t in TRAININGS if t != "published" and (c, t) in m]
        if gains and max(gains) <= 0:
            lines.append(
                "  None of them beat the published recipe on realistic + fix, and "
                "letting the teacher see the true state made things much worse. The likely "
                "cause: the teacher makes sharp decisions (descend now, close the gripper now) "
                "from facts the student cannot see, like the exact object position or where "
                "the arm is after the delay. The student can only learn the average of those "
                "decisions, and an averaged command commits to neither. The published "
                "recipe with the fix is still the best choice here.")
        elif gains:
            lines.append(f"  Best gain on realistic + fix: {max(gains):+.0f} points.")
    pub = [r for r in res["rows"] if r["training"] == "published"]
    spread = max((abs(a["success"][k] - b["success"][k]) for a in pub for b in pub
                  if a["config"] == b["config"] and a["seed"] < b["seed"] for k in conds),
                 default=0.0)
    s0 = [r for r in pub if r["config"] == "6.4 KB" and r["seed"] == 0]
    if n > 1 and s0:
        lines.append(
            "- Noise: two training seeds of a published model differ by up to "
            f"{100 * spread:.0f} points in a cell. The 6.4 KB model's 90.5% headline is seed 0 "
            f"on 200 episodes; on these {res['eval_episodes']} it scores "
            f"{_pct(s0[0]['success']['ideal'])}.")
    lines.append(END)
    return "\n".join(lines)


def update_readme(readme: Path, res: dict) -> None:
    text = readme.read_text()
    block = render(res)
    if BEGIN in text:
        a, b = text.index(BEGIN), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        at = text.index("## Not done yet")
        text = text[:at] + block + "\n\n" + text[at:]
    readme.write_text(text)


def save(res: dict, path: Path) -> None:
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
