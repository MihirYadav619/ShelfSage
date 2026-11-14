from dataclasses import dataclass, field


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    data: dict = field(default_factory=dict)


def find_book(title: str) -> SkillResult:
    """Look up which table (or shelf) currently holds the book."""
    raise NotImplementedError


def navigate(target: str) -> SkillResult:
    """Drive the base to the parking pose of a table or shelf, e.g. 'Table_B'."""
    raise NotImplementedError


def grasp(obj: str) -> SkillResult:
    """Pick up a book with the 2-finger gripper."""
    raise NotImplementedError


def place(obj: str, shelf: str) -> SkillResult:
    """Put the held book into a shelf compartment, e.g. 'S2-A3' (shelf S1/S2/S3, row A/B, column 1-4)."""
    raise NotImplementedError


def verify(obj: str, shelf: str) -> SkillResult:
    """Check with simulator ground truth that obj is inside the shelf compartment."""
    raise NotImplementedError


# The LLM planner will only be allowed to call actions listed here.
SKILLS = {
    "FIND_BOOK": find_book,
    "NAVIGATE": navigate,
    "GRASP": grasp,
    "PLACE": place,
    "VERIFY": verify,
}