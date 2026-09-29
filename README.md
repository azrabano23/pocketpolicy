# pocketpolicy

Robot AI models need a GPU. pocketpolicy shrinks a robot's control policy
until it fits on a microcontroller that costs a few dollars, and measures how
much skill survives.

## Why it matters

The SO-101 is a $120 open-source robot arm, popular for teaching and
research. AI policies like π0 are 10+ GB and need a GPU on a nearby computer.
If the skill fits on a small board wired to the arm, the arm works on its
own: no laptop, no GPU, no network.

## What we found

In simulation, on a pick-and-place task:

- The full-size policy succeeds **98.5%** of the time.
- A version that fits in **6.4 KB** still succeeds **90.5%** of the time.
- The best small version reaches **97%** in 144 KB.
- Training with corrections from the full policy (DAgger) adds 13–31 points
  to the smallest models, which drift the most on their own.

The generated C code gives exactly the same outputs as the Python model,
checked byte for byte on every run.

## Try it

```bash
pip install -e ".[test]"
pytest
pocketpolicy emit --width 64 --depth 2 --H 1 --dagger 2 -o out/   # writes the C code
```

`firmware/so101_main.c` is the control loop that would run on the arm.

## Not done yet

- The "full-size policy" is a scripted expert standing in for π0, so every
  result reproduces on a laptop. The π0 adapter is written but not run.
- The simulation has no contact physics.
- Nothing has run on a real arm yet; chip timings are estimates.

Full tables: [results/REPORT.md](results/REPORT.md).
