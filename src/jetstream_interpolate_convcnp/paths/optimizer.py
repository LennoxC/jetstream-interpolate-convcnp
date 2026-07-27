"""
optimizer.py
------------
Adam-based gradient descent over a Trajectory's control points, minimizing
the composite loss J from losses.py.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .aircraft import AircraftProfile
from .losses import LossWeights, compute_losses
from .trajectory import Trajectory
from .windfield import WindField


class TrajectoryOptimizer:
    def __init__(
        self,
        trajectory: Trajectory,
        windfield: WindField,
        aircraft: AircraftProfile,
        weights: Optional[LossWeights] = None,
        num_samples: int = 200,
        lr: float = 0.02,
    ):
        self.trajectory = trajectory
        self.windfield = windfield
        self.aircraft = aircraft
        self.weights = weights or LossWeights()
        self.num_samples = num_samples
        self.optimizer = torch.optim.Adam(trajectory.parameters(), lr=lr)
        self.history = []

    def step(self):
        self.optimizer.zero_grad()
        result = compute_losses(
            self.trajectory, self.windfield, self.aircraft, self.weights, self.num_samples
        )
        result.total.backward()
        self.optimizer.step()
        return result

    def run(
        self,
        num_iters: int = 300,
        verbose: bool = True,
        log_every: int = 25,
        callback: Optional[Callable[[int, "LossResult"], None]] = None,  # noqa: F821
        grad_clip: Optional[float] = 5.0,
    ):
        for i in range(num_iters):
            self.optimizer.zero_grad()
            result = compute_losses(
                self.trajectory, self.windfield, self.aircraft, self.weights, self.num_samples
            )
            result.total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.trajectory.parameters(), grad_clip)
            self.optimizer.step()

            stats = result.as_dict()
            self.history.append(stats)
            if verbose and (i % log_every == 0 or i == num_iters - 1):
                print(
                    f"iter {i:4d} | J={stats['total']:.3f} "
                    f"T={stats['time_s']:.1f}s F={stats['fuel_kg']:.1f}kg "
                    f"C={stats['curvature']:.4f} S={stats['smoothness']:.4f} H={stats['altitude']:.1f}"
                )
            if callback is not None:
                callback(i, result)
        return self.history

    def evaluate(self, windfield: Optional[WindField] = None, num_samples: Optional[int] = None):
        """Evaluate the current trajectory's loss without taking an optimizer
        step -- e.g. against a *different* WindField to compare datasets."""
        wf = windfield or self.windfield
        ns = num_samples or self.num_samples
        with torch.no_grad():
            return compute_losses(self.trajectory, wf, self.aircraft, self.weights, ns)
