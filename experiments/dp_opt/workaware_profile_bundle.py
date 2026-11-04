import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


MANIFEST = "workaware-profile-sources.json"


def relative_path(name):
    if not isinstance(name, str):
        raise ValueError("Profile source must stay inside its archived bundle")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Profile source must stay inside its archived bundle")
    return path


def source_files(root, fit):
    names = list(fit["source_files_sha256"])
    if "composite_profile" not in fit:
        manifest = json.loads((root / "workaware-runtime-source.json").read_text())
        names.extend("workaware-runtime-source/" + name for name in manifest["sources"])
    return tuple(dict.fromkeys(str(relative_path(name)) for name in names))


def archive_sources(root, destination, fit):
    destination.mkdir()
    for name in source_files(root, fit):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)


def anchored_supplement(legacy, supplement, destination, legacy_fit, verifier):
    from workaware_pilot import SOURCE_FILES

    if (supplement / "workaware-cadence-anchor.json").exists():
        raise ValueError("Preserve existing anchored supplements")
    raw_files = {name: hashlib.sha256((supplement / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    archive_sources(supplement, destination, {"source_files_sha256": raw_files})
    archive_sources(legacy, destination / "cadence-anchor", legacy_fit)
    (destination / "workaware-cadence-anchor.json").write_text(
        json.dumps({"schema_version": 1, "source": "cadence-anchor"}, indent=2) + "\n")
    for name, expected in raw_files.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Supplement raw changed during promotion")
    return verifier(destination)


def verify_composite(root, verifier):
    manifest = json.loads((root / MANIFEST).read_text())
    sources, required = manifest.get("sources"), manifest.get("required_profiles")
    if (manifest.get("schema_version") != 1 or not isinstance(sources, list) or len(sources) < 2
            or any(not isinstance(name, str) for name in sources) or len(set(sources)) != len(sources)
            or not isinstance(required, list) or len(set(required)) != len(required)):
        raise ValueError("Composite calibration must declare independent sources and exact coverage")
    fits, hashes, profiles = [], {MANIFEST: hashlib.sha256((root / MANIFEST).read_bytes()).hexdigest()}, {}
    for name in sources:
        directory = root / relative_path(name)
        if (directory / MANIFEST).exists():
            raise ValueError("Nested composite calibration is not supported")
        fit = verifier(directory)
        overlap = set(profiles) & set(fit["population_profiles"]["profiles"])
        if overlap:
            raise ValueError("Composite calibration cannot overwrite existing population profiles")
        profiles.update(fit["population_profiles"]["profiles"])
        for filename in source_files(directory, fit):
            hashes[name + "/" + filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        fits.append(fit)
    if set(profiles) != set(required):
        raise ValueError("Composite calibration omits or changes declared population coverage")
    legacy = fits[0]
    return {**legacy, "max_population": max(fit["max_population"] for fit in fits),
            "max_context_tokens": max(fit["max_context_tokens"] for fit in fits),
            "max_attention_tokens": max(fit["max_attention_tokens"] for fit in fits),
            "cases": sum(fit["cases"] for fit in fits), "requests": sum(fit["requests"] for fit in fits),
            "source_files_sha256": hashes, "composite_profile": manifest,
            "population_profiles": {**legacy["population_profiles"], "profiles": profiles,
                                    "source_files_sha256": hashes}}


def main():
    from verify_workaware_profile import verify_profile

    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-supplement", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    legacy_fit = verify_profile(args.legacy)
    archive_sources(args.legacy, args.output / "legacy", legacy_fit)
    if args.anchor_supplement:
        anchored_supplement(args.legacy, args.supplement, args.output / "supplement", legacy_fit, verify_profile)
    else:
        archive_sources(args.supplement, args.output / "supplement", verify_profile(args.supplement))
    manifest = {"schema_version": 1, "sources": ["legacy", "supplement"],
                "required_profiles": [f"{model}:{population}" for model in (0, 1) for population in range(2, 10)]}
    (args.output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    fit = verify_profile(args.output)
    (args.output / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps({"profiles": sorted(fit["population_profiles"]["profiles"]),
                      "source_files": len(fit["source_files_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
