"""ByzMinds metrics — Stage A entry points.

Public surface (populated in milestones 2–3):

  * ``divergence.delta_ia(pi_ind, mu_act) -> float``      KL intent → action
  * ``divergence.delta_cog(pi_ind, pi_eli) -> float``     KL intent → elicited
  * ``bbi.bbi(population, malformation_policy) -> float`` population BBI
  * ``surface_honesty.classify(message, judge) -> bool``  held-out judge
  * ``manifest_reader.read(path) -> Manifest``           parsed manifest

All four read from a Step 1 manifest written by ``byzminds-run``; no
live kernel access. Stage A computes metrics offline; live computation
is Stage B+ machinery.
"""

__version__ = "0.5.0"
