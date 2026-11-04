"""Expected bounded workflow failures, distinct from uncertain device input."""


class MeterWorkflowLimit(RuntimeError):
    """The existing window/budget is spent; no new input is authorized."""
