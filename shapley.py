from itertools import combinations, chain
from math import factorial

def shapley_value(scores):
    players = ['A', 'B', 'C']
    N = len(players)
    factorials = {i: factorial(i) for i in range(N + 1)}

    def value(subset):
        if not subset:
            return scores.get('bond', 0)
        return scores.get("".join(sorted(subset)), 0)

    shapley_values = {}
    for player in players:
        shap_value = 0
        for S in chain.from_iterable(combinations(players, r) for r in range(N)):
            if player not in S:
                weight = factorials[len(S)] * factorials[N - len(S) - 1] / factorials[N]
                shap_value += weight * (value(tuple(sorted(list(S) + [player]))) - value(S))
        shapley_values[player] = shap_value
    return shapley_values

scores = {
    'ABC': 12.1403556,
    'bond': -100.538425,
    'A': -5.0639477,
    'B': -107.85726,
    'C': -37.734801,
    'AB': -22.156332,
    'AC': -16.3544287,
    'BC': -19.775314
}

print(shapley_value(scores))
