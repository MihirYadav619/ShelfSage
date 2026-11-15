from dataclasses import dataclass, field


@dataclass
class SkillResult:
    """What every skill returns: success, a message (sent to the planner on failure) and data."""
    success: bool
    message: str = ""
    data: dict = field(default_factory=dict)
