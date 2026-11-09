"""让 tests 能 import src 包与 evals-mm 下的扁平模块（目录名带连字符，无法作为包）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals-mm"))
