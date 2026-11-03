"""One user model as several role agents; code checks decide acceptance (see team.py)."""
from .team import FAMILIES, MODES, Result, Team, Turn, check_solve, extract

__all__ = ["FAMILIES", "MODES", "Result", "Team", "Turn", "check_solve", "extract"]
