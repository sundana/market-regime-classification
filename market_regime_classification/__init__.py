"""Market regime detection baseline.

The package is intentionally dependency-light: the Gaussian HMM is implemented
directly in NumPy so filtered (causal) state probabilities are available
without relying on a library's smoothed posterior.
"""

__version__ = "0.2.0"
