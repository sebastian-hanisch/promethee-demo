"""PROMETHEE: Präferenzfunktionen, Präferenzgrade, Flüsse, Teilordnung (PROMETHEE I), gewichtete Borda-Zählung und ein Vergleichswert zu ELECTRE. Nur numpy und scipy.

Alle Funktionen arbeiten auf einer normierten Matrix N (n x m, höher besser, Min-Max) und Gewichten w. Die Differenz d = N[a, j] - N[b, j] liegt in [-1, 1]; die Schwellen
q, p und s sind Anteile der Spannweite. Präferenzgrad pi(a, b) = Summe_j w_j P_j(d_j); Plusfluss Phi+(a) = Summe_b pi(a, b) / (n - 1); Minusfluss Phi-(a) = Summe_b pi(b, a) / (n - 1);
Nettofluss Phi = Phi+ - Phi-.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

import pmt_constants as C


def preference(kind: str, d, q: float = 0.0, p: float = 1.0, s: float = 0.25) -> np.ndarray:
    """Präferenzfunktion P(d) in [0, 1]: 0 für d <= 0 (b ist nicht schlechter), monoton nicht fallend in d."""
    d = np.asarray(d, dtype=float)
    if kind == "usual":
        return (d > 0).astype(float)
    if kind == "ushape":
        return (d > q).astype(float)
    if kind == "vshape":
        return np.clip(d / p, 0.0, 1.0)
    if kind == "level":
        return np.where(d > p, 1.0, np.where(d > q, 0.5, 0.0))
    if kind == "linear":
        return np.clip((d - q) / (p - q), 0.0, 1.0)
    if kind == "gauss":
        return np.where(d > 0, 1.0 - np.exp(-(d ** 2) / (2.0 * s * s)), 0.0)
    raise ValueError(f"unbekannte Präferenzfunktion: {kind}")


def preference_degrees(N: np.ndarray, w: np.ndarray, kind: str, q: float = 0.0, p: float = 1.0, s: float = 0.25) -> np.ndarray:
    """pi[a, b] = Summe_j w_j P_j(N[a, j] - N[b, j]); die Diagonale ist 0."""
    N = np.asarray(N, dtype=float)
    D = N[:, None, :] - N[None, :, :]
    pi = (preference(kind, D, q, p, s) * np.asarray(w, dtype=float)).sum(axis=-1)
    np.fill_diagonal(pi, 0.0)
    return pi


def flows(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Phi+, Phi-, Phi) aus den Präferenzgraden; bei n = 1 lauter Nullen."""
    n = len(pi)
    if n < 2:
        z = np.zeros(n)
        return z, z.copy(), z.copy()
    plus = pi.sum(axis=1) / (n - 1)
    minus = pi.sum(axis=0) / (n - 1)
    return plus, minus, plus - minus


def promethee_one(plus: np.ndarray, minus: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, I, R): a P b, wenn Phi+(a) >= Phi+(b) und Phi-(a) <= Phi-(b) mit mindestens einer strikten Ungleichung; I: beide Flüsse gleich; R: widersprüchlich (unvergleichbar).
    Toleranz eps gegen Rundung."""
    plus = np.asarray(plus, dtype=float)
    minus = np.asarray(minus, dtype=float)
    ge = plus[:, None] >= plus[None, :] - eps
    le = minus[:, None] <= minus[None, :] + eps
    strict = (plus[:, None] > plus[None, :] + eps) | (minus[:, None] < minus[None, :] - eps)
    P = ge & le & strict
    I = (np.abs(plus[:, None] - plus[None, :]) <= eps) & (np.abs(minus[:, None] - minus[None, :]) <= eps)
    np.fill_diagonal(I, False)
    R = ~P & ~P.T & ~I
    np.fill_diagonal(R, False)
    return P, I, R


def weighted_borda(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gewichtete Borda-Zählung: Summe_j w_j (Zahl der Schlechteren - Zahl der Besseren) / (n - 1); Gleichstände zählen nicht. Gleich dem Nettofluss der gewöhnlichen Funktion."""
    N = np.asarray(N, dtype=float)
    n = len(N)
    w = np.asarray(w, dtype=float)
    worse = (N[:, None, :] > N[None, :, :]).sum(axis=1)       # [a, j]: Zahl der b mit N[a, j] > N[b, j]
    better = (N[:, None, :] < N[None, :, :]).sum(axis=1)
    return ((worse - better) / (n - 1)) @ w


def has_cycle(R: np.ndarray) -> bool:
    """Gibt es einen Zyklus (eine starke Zusammenhangskomponente mit mindestens zwei Knoten) im gerichteten Graphen R?"""
    _, labels = connected_components(np.asarray(R, dtype=bool), directed=True, connection="strong")
    return bool(np.bincount(labels).max() > 1)


def electre_reference(N: np.ndarray, w: np.ndarray, cstar: float) -> np.ndarray:
    """Vergleichswert zu ELECTRE I ohne Veto: a übertrifft b, wenn die Gewichte der Kriterien, in denen a mindestens so gut ist wie b, zusammen mindestens c* betragen."""
    N = np.asarray(N, dtype=float)
    conc = ((N[:, None, :] >= N[None, :, :]) * np.asarray(w, dtype=float)).sum(axis=-1)
    S = conc >= cstar - 1e-12
    np.fill_diagonal(S, False)
    return S


def net_flow_order(phi: np.ndarray) -> np.ndarray:
    """Alternativen von der besten zur schlechtesten nach Nettofluss; bei Gleichstand die kleinere Nummer zuerst."""
    return np.argsort(-np.asarray(phi, dtype=float), kind="stable")


def uses(kind: str, param: str) -> bool:
    """Nutzt die Präferenzfunktion den Parameter ('q', 'p' oder 's')?"""
    return param in C.KIND_PARAMS[kind]
