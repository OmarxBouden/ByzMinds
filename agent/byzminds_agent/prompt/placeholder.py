"""Minimal Step-3 prompt renderer.

Step 4 lands the full L0/L1/L2 template stack per
byzminds-template-spec.md. Step 3 uses just enough rendering to drive
the model: a single short system message describing the role + a
serialized View dump as user content. The point is to exercise the
forward pass + tool-call extraction, not to study prompt sensitivity —
that's Step 4's experiment 004 / 005.

Skeleton; concrete implementation lands in milestone 2.
"""

from __future__ import annotations
