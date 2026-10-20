import sys
from pathlib import Path

# The server runs from server/ (`uvicorn app:app`) and imports its modules by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
