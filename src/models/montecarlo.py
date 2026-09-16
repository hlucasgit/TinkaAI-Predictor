"""
Simulacion Monte Carlo para generacion de combinaciones.
"""

import random


def generar_combinacion():
    """Genera una combinacion valida de 6 numeros."""

    return sorted(random.sample(range(1, 49), 6))


def generar_simulaciones(cantidad: int = 1000):
    """Genera multiples combinaciones simuladas."""

    return [generar_combinacion() for _ in range(cantidad)]
