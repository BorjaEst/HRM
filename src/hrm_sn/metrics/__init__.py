from torchmetrics import MetricCollection

from hrm_sn.metrics.adapter import update_metrics_from_step
from hrm_sn.metrics.torchmetrics import RatioMetric
from hrm_sn.metrics.types import HaltedAgg, LossAgg, StepMetrics, TokenAgg

__all__ = [
    "RatioMetric",
    "HaltedAgg",
    "LossAgg",
    "StepMetrics",
    "TokenAgg",
    "update_metrics_from_step",
    "build_metrics",
]


def build_metrics() -> MetricCollection:
    """Construct a MetricCollection with all HRM metrics."""
    return MetricCollection(
        {
            "all/accuracy": RatioMetric(),
            "halted/rate": RatioMetric(),
            "halted/avg_steps": RatioMetric(),
            "tokens/accuracy": RatioMetric(),
            "loss/lm": RatioMetric(),
            "loss/q_halt": RatioMetric(),
            "loss/q_continue": RatioMetric(),
        },
        prefix=None,
        postfix=None,
        compute_groups=[
            [
                "all/accuracy",
                "halted/rate",
                "halted/avg_steps",
                "tokens/accuracy",
                "loss/lm",
                "loss/q_halt",
                "loss/q_continue",
            ]
        ],
    )
