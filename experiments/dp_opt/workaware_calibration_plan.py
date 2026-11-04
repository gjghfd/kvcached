import math
import random


def supplemental_plan():
    return {"schema_version": 1, "seed": 20261028, "models": [0, 1], "repeats": [0, 1],
            "output_tokens": 3072, "window_tokens": [1536, 2560], "budget_mib": 10240,
            "populations": [{"population": population, "prompt_tokens": [4096, 8192, longest]}
                            for population, longest in ((5, 16384), (6, 14336), (7, 11264))]}


def planned_cases(plan):
    if (not isinstance(plan, dict) or plan.get("schema_version") != 1
            or plan.get("models") != [0, 1] or plan.get("output_tokens") != 3072
            or plan.get("window_tokens") != [1536, 2560] or plan.get("budget_mib") != 10240
            or type(plan.get("seed")) is not int):
        raise ValueError("Calibration plan must freeze both tenants and serving controls")
    repeats, populations = plan.get("repeats"), plan.get("populations")
    if (not isinstance(repeats, list) or len(repeats) < 2
            or any(type(repeat) is not int or repeat < 0 for repeat in repeats)
            or len(set(repeats)) != len(repeats) or not isinstance(populations, list) or not populations):
        raise ValueError("Calibration plan needs independent repeats and populations")
    seen = set()
    for entry in populations:
        population, lengths = entry.get("population"), entry.get("prompt_tokens")
        if (type(population) is not int or not 2 <= population <= 9 or population in seen
                or not isinstance(lengths, list) or len(lengths) < 2
                or any(type(length) is not int or length <= 0 or length + 3072 > 131072 for length in lengths)
                or len(set(lengths)) != len(lengths)):
            raise ValueError("Calibration plan has invalid population or context coverage")
        seen.add(population)
    cases = []
    for model in plan["models"]:
        for repeat in repeats:
            paired = [{"model": model, "population": entry["population"], "rank_zero": rank_zero,
                       "prompt_tokens": length, "repeat": repeat}
                      for entry in populations for length in entry["prompt_tokens"]
                      for rank_zero in range(entry["population"] + 1)]
            random.Random(plan["seed"] + model * 100 + repeat).shuffle(paired)
            cases.extend(paired)
    return cases


def conservative_mapped_mib(case, block_size=16, page_block_capacity=64, page_bytes=96 * 1024**2,
                            pinned_mib=192):
    blocks = math.ceil((case["prompt_tokens"] + 3072) / block_size)
    pages = math.ceil(blocks / page_block_capacity)
    return [pinned_mib + count * pages * page_bytes / 1024**2
            for count in (case["rank_zero"], case["population"] - case["rank_zero"])]


def screen_plan(plan):
    cases = planned_cases(plan)
    bounds = [conservative_mapped_mib(case) for case in cases]
    if any(value > plan["budget_mib"] for bound in bounds for value in bound):
        raise ValueError("Calibration case exceeds conservative mapped budget including pinned pools")
    return {"cases": len(cases), "maximum_mapped_mib": max(max(bound) for bound in bounds),
            "status": "CPU_arithmetic_only_not_GPU_capacity_verified"}
