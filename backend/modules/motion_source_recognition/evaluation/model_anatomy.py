"""What is inside the five saved models, for the training stage of the website. Split 0, the saved models at their chosen settings; nothing is searched or chosen here.

Random Forests: the top of one real tree (depth 2, feature names and thresholds), the way four example validation windows (one per class) run through it, and the vote of each of the 300 trees
  for those windows (so that the website can replay how the forest aggregates its trees).
Logistic Regressions: the largest weights of every class, and how those weights and the probabilities of the same four example windows develop when the model is trained for 5, 10, 25 ... up to
  its saved number of iterations (refitted from scratch at each iteration limit, as in evaluation/training_curves.py; the last stage reproduces the saved model).

Run from the repository root (about 10 minutes, mostly the Logistic Regression on MiniRocket):
    nice -n 15 backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.model_anatomy
Writes outputs/evaluation/model_anatomy.json.
"""
import json
import re
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..preprocessing.config import CLASSES, EVALUATION_DIR, PREP_DIR, TRAINING, WEBSITE
from ..training.common import MODELS_DIR, load_split, score

from .training_curves import FORESTS, ITERATIONS_AT, LINEARS, validation

K, SEED, DEPTH, TOP = 0, TRAINING["seed"], WEBSITE["tree_depth_shown"], WEBSITE["top_weights_per_class"]
EXAMPLES = CLASSES


def friendly(name: str) -> str:
    """'std_ch012' -> 'std, channel 12'; other names: underscores to spaces."""
    m = re.fullmatch(r"(.+)_ch(\d+)", name)
    return f"{m.group(1).replace('_', ' ')}, channel {int(m.group(2))}" if m else name.replace("_", " ")


def feature_names(route: str, n: int) -> list:
    if route == "minirocket":
        return [f"MiniRocket #{i}" for i in range(n)]
    stats = json.loads((PREP_DIR / "features_stats_columns.json").read_text())
    physical = json.loads((PREP_DIR / "features_physical_columns.json").read_text())
    names = stats if route == "statistics" else stats + ["physical: " + p for p in physical]
    return [friendly(x) for x in names]


def example_positions(y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    return {c: int(rng.choice(np.flatnonzero(y == i))) for i, c in enumerate(CLASSES)}


def class_weights() -> dict:
    w = json.loads((PREP_DIR / "baselines.json").read_text())[f"split_{K}"]["class_weights"]
    return {c: round(float(w[c]), 2) for c in CLASSES}


def tree_excerpt(est, names: list, x_examples: dict) -> dict:
    """The top DEPTH levels of one fitted tree, and for every example the ids of the excerpt nodes on its path."""
    t = est.tree_
    dist = t.value[:, 0, :] / t.value[:, 0, :].sum(axis=1, keepdims=True)
    nodes, queue, new_id = [], [(0, 0)], {0: 0}
    while queue:
        old, depth = queue.pop(0)
        leaf = t.children_left[old] == -1
        node = {"id": new_id[old], "depth": depth, "samples": int(t.n_node_samples[old]), "dist": [round(float(v), 3) for v in dist[old]], "leaf": bool(leaf)}
        if not leaf:
            node.update({"feature": names[t.feature[old]], "threshold": round(float(t.threshold[old]), 4)})
            if depth < DEPTH:
                for side, child in (("left", t.children_left[old]), ("right", t.children_right[old])):
                    new_id[child] = len(new_id)
                    node[side] = new_id[child]
                    queue.append((child, depth + 1))
            else:
                node["cut"] = True
        nodes.append(node)
    paths = {}
    for c, x in x_examples.items():
        on_path = est.decision_path(x[None]).indices
        paths[c] = [new_id[n] for n in on_path if n in new_id]
    return {"nodes": nodes, "depth": DEPTH, "paths": paths}


def forest_anatomy(model_dir: str, route: str) -> dict:
    model = joblib.load(MODELS_DIR / model_dir / f"split_{K}" / "model.joblib")
    X, y = validation(route, K)
    pos = example_positions(y)
    x_examples = {c: X[i] for c, i in pos.items()}
    names = feature_names(route, X.shape[1])
    per_tree = {c: [[round(float(v), 3) for v in est.predict_proba(x[None])[0]] for est in model.estimators_] for c, x in x_examples.items()}
    n = X.shape[1]
    return {"kind": "forest", "settings": {"trees": model.n_estimators, "min_samples_leaf": model.min_samples_leaf, "features_per_split": int(np.sqrt(n)), "features": n, "class_weights": class_weights()},
            "tree": tree_excerpt(model.estimators_[0], names, x_examples), "examples": [{"class": c, "tree_probs": per_tree[c]} for c in EXAMPLES]}


def linear_anatomy(model_dir: str, route: str) -> dict:
    run = json.loads((MODELS_DIR / model_dir / f"split_{K}" / "run.json").read_text())
    C = run["chosen_parameters"]["C"]
    data = load_split(K, route)
    scaler = StandardScaler(copy=False).fit(data.X_train)
    X_train, X_val = scaler.transform(data.X_train, copy=False), scaler.transform(data.X_val, copy=False)
    pos = example_positions(data.y_val)
    x_ex = np.stack([X_val[pos[c]] for c in EXAMPLES])
    names = feature_names(route, X_val.shape[1])
    stages, models = [], []
    with threadpool_limits(limits=3), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for limit in ITERATIONS_AT:
            m = LogisticRegression(C=C, max_iter=limit, class_weight=data.class_weights, random_state=42).fit(X_train, data.y_train)
            used = int(m.n_iter_[0])
            stages.append({"iterations": used, "limit": limit, "macro_f1": score(data.y_val, m.predict_proba(X_val))["macro_f1"], "examples": [[round(float(v), 3) for v in p] for p in m.predict_proba(x_ex)]})
            models.append((m.coef_.copy(), m.intercept_.copy()))
            if used < limit:
                break
    final_coef, final_b = models[-1]
    chosen = sorted({int(i) for c in range(len(CLASSES)) for i in np.argsort(-np.abs(final_coef[c]))[:TOP]})
    for st, (coef, b) in zip(stages, models):
        st["weights"] = {CLASSES[c]: [round(float(coef[c][i]), 4) for i in chosen] for c in range(len(CLASSES))}
        st["bias"] = [round(float(v), 4) for v in b]
    return {"kind": "linear", "settings": {"C": C, "max_iter": 300, "features": X_val.shape[1], "class_weights": class_weights()}, "features": [names[i] for i in chosen],
            "stages": stages, "examples": list(EXAMPLES), "saved_validation_macro_f1": run["validation"]["macro_f1"], "refit_validation_macro_f1": stages[-1]["macro_f1"]}


def run() -> dict:
    out = {}
    for model_dir, route in FORESTS.items():
        out[model_dir] = forest_anatomy(model_dir, route)
        print(model_dir, "done", flush=True)
    for model_dir, route in LINEARS.items():
        out[model_dir] = linear_anatomy(model_dir, route)
        a = out[model_dir]
        print(model_dir, "done, refit macro-F1", a["refit_validation_macro_f1"], "saved", a["saved_validation_macro_f1"], flush=True)
    return out


if __name__ == "__main__":
    (EVALUATION_DIR / "model_anatomy.json").write_text(json.dumps(run()))
