# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    `nppoly.polyroots()`, najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if coef is None:
        return None
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1:
        return None
    if coef.size == 0:
        return None

    noise = np.random.random_sample(coef.shape) * 1e-10
    coef_noised = coef + noise
    roots = nppoly.polyroots(coef_noised)

    return coef_noised, roots


def frob_a(coef: np.ndarray) -> np.ndarray | None:

    if coef is None:
        return None
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1:
        return None
    if coef.size < 2:
        return None
    a_n = coef[-1]
    if a_n == 0:
        return None
    n = coef.size-1 
    F = np.zeros((n, n), dtype=float)
    for i in range(n-1):
        F[i,i+1] = 1
    F[-1, :] = -coef[:-1] / a_n
    return F

def is_nonsingular(A: np.ndarray) -> bool | None:
    if not isinstance(A, np.ndarray):
        return None
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    A = A.astype(float)
    eps = np.finfo(float).eps
    detA = np.linalg.det(A)
    if np.isclose(detA, 0.0, atol=eps, rtol=0.0):
        return False
    else:
        return True

