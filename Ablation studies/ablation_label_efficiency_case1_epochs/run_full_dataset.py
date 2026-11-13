"""Run full dataset."""

import sys
import numpy as np
import run_experiment as experiment


def check():
    ids = experiment.labels(123, 1246)
    assert len(ids) == 1246 and set(ids) == set(range(1246))
    for n in experiment.COUNTS:
        assert np.array_equal(ids[:n], experiment.labels(123, n))
    dd_rng = np.random.default_rng(3123)
    hybrid_rng = np.random.default_rng(3123)
    updates = 0
    for epoch in range(1000):
        dd = experiment.epoch_batches("dd", 1246, 123, dd_rng)
        hybrid = experiment.epoch_batches("hybrid", 1246, 123, hybrid_rng)
        assert dd == hybrid and len(dd) == 39 and len(dd[-1]) == 30
        flat = [i for b in dd for i in b]
        assert len(flat) == 1246 and set(flat) == set(range(1246))
        updates += len(dd)
    assert updates == 39000
    print(
        "PASS: full coverage, nested subsets, matched DD/Hybrid epochs, 39000 updates",
        flush=True,
    )


if __name__ == "__main__":
    check()
    if "--check" not in sys.argv:
        experiment.COUNTS.append(1246)
        original_dump = experiment.dump

        def dump(path, data):
            if path.name == "manifest.json":
                data["extension_sha256"] = experiment.sha256(__file__)
                data["extension"] = "full_training_set_1246; original test set excluded"
            original_dump(path, data)

        experiment.dump = dump
        experiment.main()
