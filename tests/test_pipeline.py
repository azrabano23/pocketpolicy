from pocketpolicy import pipeline

SMALL = {"eval_episodes": 8, "train_episodes": 8, "epochs": 3}


def test_graph_shape_and_small_run(tmp_path):
    g = pipeline.graph(tmp_path)
    assert g.free_params() == {"width", "depth", "H", "dagger", "seed", "eval_episodes",
                               "train_episodes", "epochs"}
    m, keys = pipeline.execute({"width": 8, "depth": 1, "H": 2, "dagger": 0}, 0, tmp_path,
                               **SMALL)
    assert set(keys) == {"teacher_eval", "student", "calib", "deployed", "metrics"}
    assert 0 <= m["success"] <= 1 and m["flash_kb"] > 0 and m["infer_hz"] == 30.0
    # a different chunk length reuses the teacher evaluation and calibration
    r = g.run(["metrics"], {"width": 8, "depth": 1, "H": 1, "dagger": 0, "seed": 0, **SMALL})
    cached = {t.name for t in r.trace if t.cached}
    assert {"teacher_eval", "calib"} <= cached and "student" not in cached


def test_evaluation_size_is_part_of_the_cache_key(tmp_path):
    p = {"width": 8, "depth": 1, "H": 1, "dagger": 0, "seed": 0, **SMALL}
    g = pipeline.graph(tmp_path)
    g.run(["teacher_eval"], p)
    r = g.run(["teacher_eval"], {**p, "eval_episodes": 9})
    assert not r.trace[0].cached
