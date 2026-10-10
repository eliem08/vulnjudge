"""Logistic regression implemented from scratch (pure stdlib).

Full-batch gradient descent with L2 regularization. Features are all in
[0, 1], so no standardization step is needed and weights stay directly
interpretable: contribution of feature i to a prediction is w_i * x_i.
"""

import json
import math
import random


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


class LogisticRegression:
    def __init__(self, lr=0.5, iters=3000, l2=1e-3, seed=42):
        self.lr = lr
        self.iters = iters
        self.l2 = l2
        self.seed = seed
        self.w = []
        self.b = 0.0
        self.n_features = 0

    def fit(self, X, y):
        if not X:
            raise ValueError("empty training set")
        n = len(X)
        d = len(X[0])
        self.n_features = d
        rng = random.Random(self.seed)
        self.w = [rng.gauss(0.0, 0.01) for _ in range(d)]
        self.b = 0.0
        for _ in range(self.iters):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y):
                p = sigmoid(_dot(self.w, xi) + self.b)
                err = p - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                gw[j] = gw[j] / n + self.l2 * self.w[j]
                self.w[j] -= self.lr * gw[j]
            self.b -= self.lr * (gb / n)
        return self

    def decision(self, x):
        return _dot(self.w, x) + self.b

    def predict_proba(self, X):
        single = False
        if X and not isinstance(X[0], (list, tuple)):
            X = [X]
            single = True
        out = [sigmoid(self.decision(x)) for x in X]
        return out[0] if single else out

    def contributions(self, x):
        """Per-feature signed contributions w_i * x_i (explainability)."""
        return list(zip(range(len(self.w)), [wi * xi for wi, xi in zip(self.w, x)]))

    def to_dict(self):
        return {
            "weights": self.w,
            "bias": self.b,
            "lr": self.lr,
            "iters": self.iters,
            "l2": self.l2,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(lr=d.get("lr", 0.5), iters=d.get("iters", 3000),
                l2=d.get("l2", 1e-3), seed=d.get("seed", 42))
        m.w = list(d["weights"])
        m.b = float(d["bias"])
        m.n_features = len(m.w)
        return m


def save_model(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_model(path):
    with open(path) as f:
        return json.load(f)
