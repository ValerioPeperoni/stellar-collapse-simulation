"""
conftest.py di root — garantisce che il pacchetto `collasso` sia importabile
dai test in `tests/` senza richiedere installazione (pip install -e .) o
impostazione manuale di PYTHONPATH.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
