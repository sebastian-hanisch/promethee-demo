"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def abcd():
    """A = (1; 0,6; 0,8), B = (0,4; 1; 0,7), C = (0,5; 0,5; 0,2), D = (0; 0; 0), w = (0,5; 0,3; 0,2) (Min-Max-normiert; Summe 0,84 / 0,64 / 0,44 / 0).
    usual: pi(A, B) = 0,7, pi(B, A) = 0,3, pi(B, C) = pi(C, B) = 0,5; Phi+ = 0,9 / 0,6 / 0,5 / 0, Phi- = 0,1 / 0,4 / 0,5 / 1, Phi = 0,8 / 0,2 / 0 / -1 (= gewichtete Borda-Zählung).
    V-Form p = 1: Phi = 0,48 / 0,2133 / -0,0533 / -0,64 (= (4 * Summe - 1,92) / 3). V p = 0,5: Phi = 0,6867 / 0,3333 / -0,0933 / -0,9267.
    linear q = 0,1, p = 0,5: Phi = 0,6583 / 0,3667 / -0,1167 / -0,9083. Summe der Phi ist 0; Ordnung 1 > 2 > 3 > 4."""
    return np.array([[1.0, 0.6, 0.8], [0.4, 1.0, 0.7], [0.5, 0.5, 0.2], [0.0, 0.0, 0.0]]), np.array([0.5, 0.3, 0.2])


@pytest.fixture
def xyz():
    """X = (1; 0), Y = (0; 1), Z = (0,6; 0,6), w = (0,5; 0,5): usual Phi = 0 / 0 / 0 (Gleichstand, jeder Zähler 1:1); V-Form p = 1: Phi = -0,05 / -0,05 / 0,10 (Z gewinnt)."""
    return np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]]), np.array([0.5, 0.5])


@pytest.fixture
def incomparable():
    """X1 = (1; 0; 0,5), X2 = (0; 1; 0,5), X3 = (0,5; 0,5; 0), X4 = (0,2; 0,2; 1), w = (0,4; 0,4; 0,2), V-Form p = 0,5:
    Phi+ = 0,4667 / 0,4667 / 0,4267 / 0,3067, Phi- = 0,3867 / 0,3867 / 0,4667 / 0,4267, Phi = 0,08 / 0,08 / -0,04 / -0,12.
    X1 I X2 (beide Flüsse gleich); X1 und X2 P X3 und P X4; X3 R X4 (Phi+(X3) > Phi+(X4), aber Phi-(X3) > Phi-(X4))."""
    return np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 0.0], [0.2, 0.2, 1.0]]), np.array([0.4, 0.4, 0.2])
