"""Simulate the CI minimum-dependency boundary without changing installed packages."""
import importlib.abc
from pathlib import Path
import sys
import pytest


class MissingStudy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'bs4' or fullname.startswith('bs4.'):
            raise ModuleNotFoundError("No module named 'bs4'", name='bs4')


sys.meta_path.insert(0, MissingStudy())
sys.path.insert(0, str(Path.cwd()))
raise SystemExit(pytest.main(['-q', 'tests/test_frozen_cycle_resume.py']))
