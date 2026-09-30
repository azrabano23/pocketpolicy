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
pocketpolicy sim2real --seeds 2 --jobs 4   # the sim-to-real table below (~20 min, 4 cores)
```

`firmware/so101_main.c` is the control loop that would run on the arm.

<!-- sim2real:begin -->
## Sim-to-real: what survives

The simulator hands the policy perfect numbers. A real arm will not. We fed the policy corrupted readings while the physics ran on the true state, 400 held-out episodes per cell, int8 pipeline, mean of 2 training seeds (`pocketpolicy sim2real`, `results/sim2real.json`).

| policy | training | ideal | ideal + fix | sag 1°, no fix | sag 1° + fix | realistic, no fix | realistic + fix | pessimistic + fix |
|---|---|---|---|---|---|---|---|---|
| teacher | scripted | 99% | 95% | 0% | 94% | 0% | 61% | 0% |
| 6.4 KB | published | 83% | 84% | 0% | 83% | 0% | 52% | 8% |
| 6.4 KB | robust, teacher sees truth | 5% | 4% | 0% | 3% | 0% | 4% | 1% |
| 6.4 KB | robust, teacher sees truth at reading | 42% | 40% | 0% | 41% | 0% | 30% | 5% |
| 6.4 KB | robust, teacher sees reading | 76% | 71% | 0% | 72% | 0% | 49% | 5% |
| 144 KB | published | 96% | 96% | 2% | 96% | 2% | 83% | 21% |
| 144 KB | robust, teacher sees truth | 30% | 30% | 0% | 30% | 0% | 29% | 10% |
| 144 KB | robust, teacher sees truth at reading | 69% | 70% | 0% | 69% | 0% | 48% | 11% |
| 144 KB | robust, teacher sees reading | 97% | 96% | 26% | 97% | 2% | 80% | 7% |

- **realistic**: detector noise 3 mm and bias 5 mm, 2 ticks of latency, 1.5° joint calibration error, 1° gravity sag, a 1.5 cm grasp window, grasp flag wrong 2% of ticks. **pessimistic**: 5 mm, 10 mm, 3 ticks, 3°, 2°, 1.0 cm, 5%. These magnitudes are guesses, not measurements.
- **Gravity sag broke the original control loop.** The policy sent `measured position + step`. A servo holding weight stops a little short of its target, so the arm stalled short of every object: with 1° of sag, success fell to 0% (6.4 KB), 2% (144 KB) and 0% for the teacher. The fix: add the step to the last command sent, not to the reading. That brings it back to 83% and 96%. `firmware/so101_main.c` now does this (`firmware/cmd_relative.c`), and a test checks that the C and Python versions produce the same numbers.
- **The big model holds up better.** With realistic errors and the fix: 83% for 144 KB, 52% for 6.4 KB. With pessimistic errors: 21% and 8%.
- **Training on noisy readings (domain-randomised DAgger).** Each training episode got its own random errors, from none up to 1.25x realistic, and ran with the fix. The student learned from what it saw. We tried three choices of what the teacher sees when it labels. Points gained over the published model (6.4 KB / 144 KB):
  - robust, teacher sees truth: ideal -78 / -66, realistic + fix -48 / -54, pessimistic + fix -7 / -11.
  - robust, teacher sees truth at reading: ideal -40 / -27, realistic + fix -22 / -35, pessimistic + fix -3 / -10.
  - robust, teacher sees reading: ideal -7 / +1, realistic + fix -3 / -3, pessimistic + fix -3 / -14.
  None of them beat the published recipe on realistic + fix, and letting the teacher see the true state made things much worse. The likely cause: the teacher makes sharp decisions (descend now, close the gripper now) from facts the student cannot see, like the exact object position or where the arm is after the delay. The student can only learn the average of those decisions, and an averaged command commits to neither. The published recipe with the fix is still the best choice here.
- Noise: two training seeds of a published model differ by up to 9 points in a cell. The 6.4 KB model's 90.5% headline is seed 0 on 200 episodes; on these 400 it scores 85%.
<!-- sim2real:end -->

## Not done yet

- The "full-size policy" is a scripted expert standing in for π0, so every
  result reproduces on a laptop. The π0 adapter is written but not run.
- The simulation has no contact physics.
- Nothing has run on a real arm yet; chip timings are estimates.
- The sim-to-real error sizes are assumed, not measured on an SO-101. The
  firmware's `CMD_LAT` must be measured on the arm too.

Full tables: [results/REPORT.md](results/REPORT.md).
