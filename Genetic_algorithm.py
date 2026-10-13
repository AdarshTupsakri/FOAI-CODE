"""
Feature Selection using a Genetic Algorithm
=============================================

A from-scratch genetic algorithm (bit-string encoding, tournament selection,
single-point crossover, bit-flip mutation, elitism) that searches for the
feature subset which maximizes cross-validated classification accuracy,
with a small penalty for subset size.

Dataset: Breast Cancer Wisconsin (sklearn built-in), 30 numeric features,
569 samples, binary classification (malignant / benign).

Compared against:
  - All features (baseline)
  - Filter method: SelectKBest (ANOVA F-value), same k as the GA finds
  - Genetic algorithm (wrapper method, this script)

Run: python3 ga_feature_selection.py
Writes: ga_results.json  (used to populate the slide deck and the web demo)
"""

import json
import random
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------- data ----
data = load_breast_cancer()
X, y = data.data, data.target
FEATURE_NAMES = list(data.feature_names)
N_FEATURES = X.shape[1]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
    ])


def cv_accuracy(mask):
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return 0.0
    scores = cross_val_score(make_model(), X[:, idx], y, cv=CV, scoring="accuracy")
    return float(scores.mean())


# ----------------------------------------------------------- GA config ----
POP_SIZE = 30
N_GENERATIONS = 50
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 1.0 / N_FEATURES
ELITISM = 2
ALPHA = 0.9  # weight on accuracy vs. (1 - weight) on compactness


def fitness(mask, cache):
    key = mask.tobytes()
    if key not in cache:
        acc = cv_accuracy(mask)
        k = int(mask.sum())
        cache[key] = ALPHA * acc + (1 - ALPHA) * (1 - k / N_FEATURES)
    return cache[key]


def random_individual():
    ind = np.random.randint(0, 2, size=N_FEATURES)
    if ind.sum() == 0:
        ind[np.random.randint(N_FEATURES)] = 1
    return ind


def tournament_select(pop, fits):
    idxs = np.random.choice(len(pop), TOURNAMENT_K, replace=False)
    best = max(idxs, key=lambda i: fits[i])
    return pop[best].copy()


def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = random.randint(1, N_FEATURES - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutate(ind):
    for i in range(N_FEATURES):
        if random.random() < MUTATION_RATE:
            ind[i] = 1 - ind[i]
    if ind.sum() == 0:
        ind[np.random.randint(N_FEATURES)] = 1
    return ind


def run_ga():
    cache = {}
    population = [random_individual() for _ in range(POP_SIZE)]
    history = []
    best_ever = None
    best_ever_fit = -1

    for gen in range(N_GENERATIONS + 1):
        fits = [fitness(ind, cache) for ind in population]
        gen_best_i = int(np.argmax(fits))
        gen_best = population[gen_best_i]
        gen_best_fit = fits[gen_best_i]
        if gen_best_fit > best_ever_fit:
            best_ever_fit = gen_best_fit
            best_ever = gen_best.copy()

        best_acc = cv_accuracy(best_ever)
        history.append({
            "generation": gen,
            "best_fitness": round(float(best_ever_fit), 4),
            "mean_fitness": round(float(np.mean(fits)), 4),
            "best_accuracy": round(float(best_acc), 4),
            "best_k": int(best_ever.sum()),
        })

        if gen == N_GENERATIONS:
            break

        new_pop = []
        elite_idx = np.argsort(fits)[-ELITISM:]
        for i in elite_idx:
            new_pop.append(population[i].copy())
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        population = new_pop

    return best_ever, history, cache


def main():
    t0 = time.time()

    all_mask = np.ones(N_FEATURES, dtype=int)
    acc_all = cv_accuracy(all_mask)

    best_mask, history, cache = run_ga()
    ga_k = int(best_mask.sum())
    ga_acc = cv_accuracy(best_mask)
    ga_features = [FEATURE_NAMES[i] for i in range(N_FEATURES) if best_mask[i] == 1]

    selector = SelectKBest(f_classif, k=ga_k)
    selector.fit(X, y)
    filter_mask = np.zeros(N_FEATURES, dtype=int)
    filter_mask[selector.get_support(indices=True)] = 1
    acc_filter = cv_accuracy(filter_mask)

    elapsed = time.time() - t0

    results = {
        "dataset": "Breast Cancer Wisconsin (sklearn)",
        "n_samples": int(X.shape[0]),
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "model": "Logistic Regression (StandardScaler + 5-fold CV)",
        "ga_config": {
            "population_size": POP_SIZE,
            "generations": N_GENERATIONS,
            "tournament_k": TOURNAMENT_K,
            "crossover_rate": CROSSOVER_RATE,
            "mutation_rate": round(MUTATION_RATE, 4),
            "elitism": ELITISM,
            "alpha": ALPHA,
        },
        "accuracy": {
            "all_features": round(acc_all * 100, 1),
            "filter_selectkbest": round(acc_filter * 100, 1),
            "genetic_algorithm": round(ga_acc * 100, 1),
        },
        "n_features_selected": {
            "all_features": N_FEATURES,
            "filter_selectkbest": ga_k,
            "genetic_algorithm": ga_k,
        },
        "ga_selected_features": ga_features,
        "convergence": history,
        "fitness_evaluations_cached": len(cache),
        "runtime_seconds": round(elapsed, 1),
    }

    with open("ga_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()