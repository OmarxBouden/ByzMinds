"""ByzMinds analysis package — Step 5.

Offline metrics over completed manifests. The package is intentionally
separate from ``byzminds_agent``: agent code runs the experiments and
writes manifests; analysis code reads manifests and computes the
substrate-paper metrics. No gRPC, no model loading, no kernel calls.
"""
