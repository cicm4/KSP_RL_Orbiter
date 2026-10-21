"""Exact memoization of pure tensor functions for the batched simulators.

The batched environments recompute the same telemetry several times per
control step (the step itself, the shaping potential, the observation, the
next step and the training loop's reset). A ``TensorMemo`` returns the stored
result only while every declared input is the *same tensor object* with the
*same in-place version counter* and every declared scalar compares equal.
Any reassignment (``env.r = ...``) or in-place write (``env.r[ids] = ...``,
``env.state += ...``) therefore invalidates it, and a hit returns exactly the
values the uncached computation would produce. Cached outputs are also
version-checked, so a caller mutating a returned tensor forces recomputation.

Inference-mode tensors carry no version counter and are never cached, and
nothing is cached while autograd could record a graph through the inputs.
"""
from __future__ import annotations

import torch


class TensorMemo:
    __slots__ = ("_inputs", "_versions", "_scalars", "_outputs", "_output_versions", "_result")

    def __init__(self):
        self.clear()

    def clear(self):
        self._inputs = self._versions = self._scalars = None
        self._outputs = self._output_versions = self._result = None

    @staticmethod
    def _cacheable(tensors):
        grad = torch.is_grad_enabled()
        for tensor in tensors:
            if tensor.is_inference() or (grad and tensor.requires_grad):
                return False
        return True

    def get(self, tensors, scalars, compute):
        """Return ``compute()``, reusing the previous result when exact.

        ``compute`` must be a pure function of ``tensors`` and ``scalars``.
        Dict results are returned as a fresh shallow copy on every call.
        """
        tensors = tuple(tensors)
        if not self._cacheable(tensors):
            self.clear()
            return compute()
        inputs = self._inputs
        if (inputs is not None and len(inputs) == len(tensors)
                and all(a is b for a, b in zip(inputs, tensors))
                and self._versions == tuple(t._version for t in tensors)
                and self._scalars == scalars
                and all(o._version == v for o, v in zip(self._outputs, self._output_versions))):
            result = self._result
        else:
            result = compute()
            outputs = tuple(result.values()) if isinstance(result, dict) else (result,)
            if not all(isinstance(o, torch.Tensor) for o in outputs) or not self._cacheable(outputs):
                self.clear()
                return result
            self._inputs, self._versions = tensors, tuple(t._version for t in tensors)
            self._scalars, self._result = scalars, result
            self._outputs = outputs
            self._output_versions = tuple(o._version for o in outputs)
        return dict(result) if isinstance(result, dict) else result
