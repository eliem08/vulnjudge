"""VulnJudge — calibrated tiny decision model for bug-bounty vulnerability triage."""

__version__ = "0.1.0"
__author__ = "Ilyass Emssaad"

from .model import LogisticRegression  # noqa: F401
from .features import FEATURE_NAMES, extract_features, fit_class_priors  # noqa: F401
