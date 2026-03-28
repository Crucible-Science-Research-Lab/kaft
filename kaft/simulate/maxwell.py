"""
MaxwellDynamics — electromagnetic field evolution on a 2D manifold.

∂B/∂t = -∇×E
∂E/∂t =  ∇×B  (vacuum, no sources)

Yee FDTD scheme. The manifold carries the field state.
"""

from __future__ import annotations
import numpy as np
from kaft.simulate.base import AbstractManifoldDynamics


class MaxwellDynamics(AbstractManifoldDynamics):
    """
    Manifold dict must contain:
        'Ex': np.ndarray (ny, nx)
        'Ey': np.ndarray (ny, nx)
        'Bz': np.ndarray (ny, nx)
        'dx': float  (default 1.0)
        'dy': float  (default 1.0)
    """

    def run(self, manifold: dict, dt: float = 0.5) -> dict:
        Ex = manifold['Ex'].copy()
        Ey = manifold['Ey'].copy()
        Bz = manifold['Bz'].copy()
        dx = manifold.get('dx', 1.0)
        dy = manifold.get('dy', 1.0)

        # B update — curl of E
        dEy_dx = (Ey[:, 1:] - Ey[:, :-1]) / dx
        dEx_dy = (Ex[1:, :] - Ex[:-1, :]) / dy
        Bz[:-1, :-1] -= dt * (dEy_dx[:-1, :] - dEx_dy[:, :-1])

        # E update — curl of B
        dBz_dy = (Bz[1:, :] - Bz[:-1, :]) / dy
        dBz_dx = (Bz[:, 1:] - Bz[:, :-1]) / dx
        Ex[1:-1, :] += dt * dBz_dy[:-1, :]
        Ey[:, 1:-1] -= dt * dBz_dx[:, :-1]

        return {**manifold, 'Ex': Ex, 'Ey': Ey, 'Bz': Bz}

    def geometry_type(self) -> str:
        return 'flat'