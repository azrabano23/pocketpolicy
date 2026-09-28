"""The bit-exact gate for the whole deployed pipeline, prep through post."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from loopgraph.cgate import compile_and_run

from .deploy import DeployedPolicy


@dataclass
class PipelineExact:
    n: int
    mismatched: int
    digest: str

    @property
    def ok(self) -> bool:
        return self.mismatched == 0


def check_c(policy: DeployedPolicy, raw: np.ndarray, name: str = "pocket") -> PipelineExact:
    raw = np.atleast_2d(np.asarray(raw, np.int16))
    ref = policy.step_int(raw)
    up = name.upper()
    main = f"""#include <stdio.h>
#include "{name}.h"
int main(void) {{
    int16_t raw[{up}_IN], out[{up}_OUT];
    int v;
    for (;;) {{
        for (int i = 0; i < {up}_IN; ++i) {{
            if (scanf("%d", &v) != 1) return 0;
            raw[i] = (int16_t)v;
        }}
        {name}_step(raw, out);
        for (int j = 0; j < {up}_OUT; ++j) printf("%d ", out[j]);
        printf("\\n");
    }}
}}
"""
    stdin = "\n".join(" ".join(str(int(v)) for v in r) for r in raw) + "\n"
    out = compile_and_run(policy.emit(name), main, stdin)
    got = np.array([[int(t) for t in l.split()] for l in out.strip().splitlines()],
                   np.int16).reshape(ref.shape)
    bad = int(np.sum(np.any(got != ref, axis=1)))
    return PipelineExact(len(raw), bad, hashlib.sha256(got.tobytes()).hexdigest()[:16])
