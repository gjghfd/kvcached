import math


def capacity_setting(samples, code_marks, page_mib=96, guard_mib=1024, maximum_budget_mib=14336,
                     allocator_gpu_utilization=1.0):
    if (not samples or not code_marks or min(page_mib, guard_mib, maximum_budget_mib) <= 0
            or any(not math.isfinite(value) for value in (page_mib, guard_mib, maximum_budget_mib))
            or not 0 < allocator_gpu_utilization <= 1):
        raise ValueError("Require measured memory, Code marks and positive capacity limits")
    headrooms = []
    for sample in samples:
        if len(sample["gpu_memory"]) != 2 or len(sample["mapped_mib"]) != 2:
            raise ValueError("Capacity selection requires two GPUs")
        for memory, mapped in zip(sample["gpu_memory"], sample["mapped_mib"]):
            if not 0 <= memory["free_mib"] <= memory["total_mib"] or mapped < 0:
                raise ValueError("Invalid GPU memory observation")
            effective_guard = max(guard_mib, memory["total_mib"] * (1 - allocator_gpu_utilization))
            headrooms.append(memory["free_mib"] + mapped - effective_guard)
    budget = int(math.floor(min(maximum_budget_mib, *headrooms) / page_mib) * page_mib)
    code_mib = max(math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / 16) * 1.5
                   for mark in code_marks)
    if budget < math.ceil(code_mib / page_mib) * page_mib + 3 * page_mib:
        raise MemoryError("Measured A800 headroom cannot admit the longest 100K Code with page/guard margin")
    lower, upper = 2 * (budget - code_mib), 2 * budget - code_mib
    target = lower + min(code_mib * .1, 768)
    return {"kv_budget_mib_per_gpu": budget, "target_conv_reserved_mib": target,
            "longest_code_logical_mib": code_mib, "ideal_conv_fragmentation_window_mib": [lower, upper],
            "non_KV_guard_mib": guard_mib, "page_bundle_mib": page_mib,
            "allocator_gpu_utilization": allocator_gpu_utilization,
            "effective_non_KV_guard_mib_per_gpu": [
                max(guard_mib, max(sample["gpu_memory"][rank]["total_mib"] for sample in samples)
                    * (1 - allocator_gpu_utilization)) for rank in (0, 1)],
            "maximum_budget_mib": maximum_budget_mib,
            "target_offset_above_local_blocking_boundary_mib": target - lower,
            "scope": "finite_sample_headroom_minus_guard_not_peak_certificate_or_placement_benefit"}


def calibration_rates(setting, conv_marks, measured_service_s, pilot_output_tokens):
    if (measured_service_s <= 0 or pilot_output_tokens <= 0 or not conv_marks
            or not math.isfinite(measured_service_s)
            or any(min(mark["prompt_tokens"], mark["output_tokens"], mark.get("count", 1)) <= 0 for mark in conv_marks)):
        raise ValueError("Require positive measured Conv service and paired marks")
    count = sum(mark.get("count", 1) for mark in conv_marks)
    memory_output = sum(math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / 16) * 1.5
                        * mark["output_tokens"] * mark.get("count", 1) for mark in conv_marks) / count
    central = setting["target_conv_reserved_mib"] * pilot_output_tokens / (memory_output * measured_service_s)
    rates = sorted({max(.05, round(central * factor, 2)) for factor in (.5, .8, 1.2, 1.6)})
    return {"conv_rates_rps": rates, "central_conv_rate_rps": central,
            "multipliers": [.5, .8, 1.2, 1.6], "measured_conv_service_s": measured_service_s,
            "pilot_output_tokens": pilot_output_tokens,
            "scope": "capacity_pilot_initial_bracket_then_open_loop_validation_not_sustainable_rate"}
