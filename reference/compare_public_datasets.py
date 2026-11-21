"""Compare the wheel and Rust-backed estimators on bundled measurement data."""

import argparse
import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import warnings

if platform.system() == "Linux" and platform.machine() == "x86_64":
    os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4,AVX512_ICL,AVX512_SPR"

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def record(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [record(item) for item in value]
    array = np.asarray(value)
    dtype = array.dtype.name if array.dtype.names is None else [list(field) for field in array.dtype.descr]
    return {"dtype": dtype, "shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def observe(call):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = {"result": record(call())}
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
            result = {"error": [type(error).__name__, str(error)]}
    return {**result, "warnings": [[item.category.__name__, str(item.message)] for item in captured]}


def fitted_outputs(model, points):
    model.fit(points)
    return (
        model.labels_, model.probabilities_, model.cluster_persistence_, model.minimum_spanning_tree_.to_numpy(),
        model.single_linkage_tree_.to_numpy(), model.condensed_tree_.to_numpy(), model.outlier_scores_,
        model.relative_validity_,
    )


def dataset_comparison(module):
    results = {}
    for load in (load_iris, load_wine, load_breast_cancer, load_digits):
        points = StandardScaler().fit_transform(load().data)
        dataset = {"input": record(points), "calls": {}}
        results[load.__name__] = dataset
        for metric in ("euclidean", "manhattan", "chebyshev", "minkowski"):
            parameters = {"p": 3.0} if metric == "minkowski" else {}
            prediction = metric in ("euclidean", "manhattan")
            for algorithm in ("best", "generic", "prims_balltree", "boruvka_balltree"):
                model = module.HDBSCAN(
                    metric=metric, algorithm=algorithm, min_samples=5, min_cluster_size=11,
                    gen_min_span_tree=True, core_dist_n_jobs=1, prediction_data=prediction, **parameters,
                )
                name = f"{metric}/{algorithm}"
                calls = dataset["calls"]
                calls[f"fit/{name}"] = observe(lambda: fitted_outputs(model, points))
                if prediction and "result" in calls[f"fit/{name}"]:
                    calls[f"predict/{name}"] = observe(
                        lambda: module.approximate_predict(model, points[:40], return_connecting_points=True)
                    )
                    calls[f"scores/{name}"] = observe(lambda: module.approximate_predict_scores(model, points[:40]))
                if load is not load_digits:
                    calls[f"robust/{name}"] = observe(lambda: module.robust_single_linkage(
                        points, 2.5, k=5, gamma=8, metric=metric, algorithm=algorithm, **parameters,
                    ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("hdbscan", "hdbscan_alternative"), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        cases = dataset_comparison(importlib.import_module(args.module))
    if args.output:
        args.output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8", newline="\n")
        return
    expected = json.loads(args.check.read_text(encoding="utf-8"))
    for name, data in cases.items():
        if data["input"] != expected[name]["input"]:
            parser.exit(1, f"{name}: input differs\n")
        for call, result in data["calls"].items():
            if result != expected[name]["calls"].get(call):
                parser.exit(1, f"{name}: {call} differs\n")
    if cases != expected:
        parser.exit(1, "Recorded calls differ\n")
    total = sum(len(data["calls"]) for data in cases.values())
    print(f"{total} public dataset calls matched {args.check}")


if __name__ == "__main__":
    main()
