"""Render the migration-only paper figures into Paper/figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migration_figures import main

if __name__ == "__main__":
    main(sys.argv[1:])
