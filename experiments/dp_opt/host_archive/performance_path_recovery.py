import hashlib
import sys
from pathlib import Path


SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    original_directory = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(original_directory))
    import performance_path_recovery as original
    import performance_archive_bridge as bridge

    return bridge.run_entry(original, Path(__file__).resolve(), SOURCE_SHA256_AT_IMPORT, sys.argv[1:])


if __name__ == "__main__":
    main()
