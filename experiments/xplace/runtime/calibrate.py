import asyncio
import random
import statistics
import time

from controller import GIB, used_gib
from engine import save, stream

KV_BYTES_PER_TOKEN = 98304


def synthetic_prompt(tokens, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(tokens)]


def window_rate(result, start, end):
    points = [(time_s, count) for time_s, count in result["cadence"] if start <= time_s <= end]
    if len(points) < 2 or points[-1][1] <= points[0][1]:
        return None
    return (points[-1][0] - points[0][0]) / (points[-1][1] - points[0][1])


def steady_step(results, margin_s=1.0):
    """Median per-token time while every request in the case is decoding."""
    start = max(result["first_token_at"] for result in results) + margin_s
    end = min(result["completed_at"] for result in results) - margin_s
    rates = [rate for rate in (window_rate(result, start, end) for result in results) if rate is not None]
    if end - start < 3 or not rates:
        return None, {"window_s": end - start, "samples": len(rates)}
    return statistics.median(rates), {"window_s": end - start, "samples": len(rates),
                                      "spread": [min(rates), max(rates)]}


def internal_ids(observed, model, external_ids):
    mapping = {}
    for rank, observation in enumerate(observed[model]):
        for request in observation["requests"]:
            base = request["id"].rsplit("-", 1)[0] if request["id"] not in external_ids else request["id"]
            if base in external_ids:
                mapping[base] = (request["id"], rank)
    return mapping


async def wait_decoding(controller, model, expected, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summaries = await controller.summaries()
        records = [request for rank in summaries[model] for request in rank["requests"]]
        if len(records) == expected and all(request["decoding"] for request in records):
            return summaries
        await asyncio.sleep(.25)
    raise TimeoutError("Calibration requests did not all reach decode")


def fits_budget(controller, summaries, model, layout, contexts, output_tokens, margin_gib=0.5):
    used = used_gib(summaries)
    for rank, count in enumerate(layout):
        need = sum(contexts[rank][index] + output_tokens for index in range(count)) * KV_BYTES_PER_TOKEN / GIB
        if used[rank] + need > controller.budgets[rank] - margin_gib:
            return False
    return True


async def measure_case(engines, controller, case, seed):
    """Run one placement case: case = {name, model, layout[rank] counts, contexts[rank][i], output_tokens}."""
    model, layout, contexts, output = case["model"], case["layout"], case["contexts"], case["output_tokens"]
    summaries = await controller.summaries()
    if not fits_budget(controller, summaries, model, layout, contexts, output):
        return {**case, "status": "skipped_budget"}
    started = time.monotonic()
    tasks = [asyncio.create_task(stream(engines[model], f"cal_{case['name']}_{rank}_{index}",
                                        synthetic_prompt(contexts[rank][index], seed + 7919 * rank + index),
                                        output, rank, model, "calibration", time.monotonic()))
             for rank, count in enumerate(layout) for index in range(count)]
    results = await asyncio.gather(*tasks)
    step, window = steady_step(results)
    return {**case, "status": "measured", "step_s": step, "window": window, "wall_s": time.monotonic() - started,
            "first_token_spread_s": max(r["first_token_at"] for r in results) - min(r["first_token_at"] for r in results)}


def conv_grid(num_ranks, output_tokens=1024):
    cases = []

    def add(name, layout, context):
        contexts = [[context] * count for count in layout]
        cases.append({"name": name, "model": 1, "layout": layout, "contexts": contexts, "output_tokens": output_tokens,
                      "max_count": max(layout), "max_context": max(sum(row) for row in contexts)})

    for count in (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32):
        add(f"bal{count}_c4k", [count] * num_ranks, 4096)
    for count in (8, 16):
        for delta in (2, 4, 8):
            add(f"skew{count}d{delta}_c4k", [count + delta, count, count, count - delta][:num_ranks], 4096)
        three = -(-count * num_ranks // (num_ranks - 1))
        add(f"empty{count}_c4k", [0] + [three] * (num_ranks - 1), 4096)
    for context in (1024, 8192, 16384):
        add(f"bal8_c{context // 1024}k", [8] * num_ranks, context)
    skewed = {"name": "ctxskew8", "model": 1, "layout": [8] * num_ranks,
              "contexts": [[16384] * 8] + [[2048] * 8 for _ in range(num_ranks - 1)], "output_tokens": output_tokens}
    skewed.update(max_count=8, max_context=16384 * 8)
    cases.append(skewed)
    return cases


async def run_conv_grid(engines, controller, root, num_ranks, seed=11):
    records = []
    for case in conv_grid(num_ranks):
        await wait_idle(controller)
        record = await measure_case(engines, controller, case, seed)
        records.append({key: value for key, value in record.items() if key != "contexts"})
        save(root / "conv-grid.json", records)
    return records


async def wait_idle(controller, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summaries = await controller.summaries()
        if all(not rank["requests"] for ranks in summaries for rank in ranks):
            return
        await asyncio.sleep(.5)
    raise TimeoutError("Engines did not become idle")


async def run_interference(engines, controller, root, code_prompt, num_ranks, conv_counts=(0, 2, 8, 16), seed=23):
    """Code (rank 0) alone and with balanced Conv background; Conv rate measured during Code prefill/decode."""
    layouts = [(f"bal{count}", [count] * num_ranks) for count in conv_counts]
    return await run_interference_layouts(engines, controller, root, code_prompt, layouts, seed=seed,
                                          output_name="interference.json")


INTERFERENCE_LAYOUTS = [(name, layout) for count, isolated in ((4, [0, 5, 5, 6]), (8, [0, 11, 11, 10]),
                                                                (12, [0, 16, 16, 16]), (16, [0, 21, 21, 22]))
                        for name, layout in ((f"bal{count}", [count] * 4), (f"iso{count}", isolated))]


async def run_interference_layouts(engines, controller, root, code_prompt, layouts, seed=23, repeats=1,
                                   output_name="interference-layouts.json", conv_output=3000):
    """Code on rank 0 with Conv layouts (per-rank counts); does Conv on the Code's GPU drive interference?"""
    records = []
    for repeat in range(repeats):
        for name, layout in layouts:
            await wait_idle(controller)
            conv_tasks = [asyncio.create_task(stream(engines[1], f"if_{name}_{repeat}_{rank}_{index}",
                                                     synthetic_prompt(4096, seed + 1009 * repeat + 101 * rank + index),
                                                     conv_output, rank, 1, "interference", time.monotonic()))
                          for rank, count in enumerate(layout) for index in range(count)]
            if conv_tasks:
                await wait_decoding(controller, 1, len(conv_tasks))
                await asyncio.sleep(3)
            code_started = time.monotonic()
            code = await stream(engines[0], f"if_code_{name}_{repeat}", code_prompt, 64, 0, 0, "interference",
                                code_started)
            conv = await asyncio.gather(*conv_tasks)
            record = {"name": name, "repeat": repeat, "layout": layout, "conv_per_rank": max(layout),
                      "conv_on_code_gpu": layout[0], "code_prefill_s": code["first_token_at"] - code["submitted_at"],
                      "code_tpot_s": (code["completed_at"] - code["first_token_at"]) / 63}
            if conv:
                def median_rate(start, end):
                    rates = [rate for rate in (window_rate(result, start, end) for result in conv) if rate is not None]
                    return statistics.median(rates) if rates else None

                record.update(conv_before_s=median_rate(code_started - 3, code_started),
                              conv_during_prefill_s=median_rate(code["submitted_at"] + 1, code["first_token_at"] - 1),
                              conv_during_decode_s=median_rate(code["first_token_at"], code["completed_at"]),
                              conv_after_s=median_rate(code["completed_at"] + 2,
                                                       min(result["completed_at"] for result in conv) - 1))
            records.append(record)
            save(root / output_name, records)
    return records


async def run_dataplane(engines, controller, root, num_ranks, seed=31):
    """Migrate Conv requests across several rank pairs and back; streams must finish with exact lengths."""
    await wait_idle(controller)
    names = [f"dp_{rank}_{index}" for rank in range(num_ranks) for index in range(2)]
    tasks = [asyncio.create_task(stream(engines[1], name, synthetic_prompt(4096, seed + position), 2500,
                                        int(name.split("_")[1]), 1, "dataplane", time.monotonic()))
             for position, name in enumerate(names)]
    await wait_decoding(controller, 1, len(names))
    pairs = [("dp_0_0", 1), ("dp_0_1", num_ranks - 1), ("dp_2_0", 1), (f"dp_{num_ranks - 1}_0", 0)]
    reports = []
    for direction in ("out", "back"):
        def plan_fn(observed, direction=direction):
            mapping = internal_ids(observed, 1, {name for name, _ in pairs})
            missing = [name for name, _ in pairs if name not in mapping]
            if missing:
                raise RuntimeError(f"Data-plane requests finished before the {direction} transaction: {missing}")
            moves = []
            for name, destination in pairs:
                internal, rank = mapping[name]
                target = destination if direction == "out" else int(name.split("_")[1])
                if rank != target:
                    moves.append({"id": internal, "model": 1, "source": rank, "destination": target})
            return {"moves": moves}

        report = await controller.run_transaction(plan_fn, f"dataplane_{direction}",
                                                  pause_models=(0, 1) if direction == "out" else (1,))
        reports.append({key: value for key, value in report.items() if key != "observations"})
        if report["status"] != "completed" or report.get("move_failure") or len(report["moves"]) != len(pairs):
            raise RuntimeError(f"Data-plane transaction {direction} failed: {report.get('status')} "
                               f"{report.get('move_failure')} moves={len(report['moves'])}")
        await asyncio.sleep(.5)
    results = await asyncio.gather(*tasks)
    summary = {"status": "passed", "transactions": reports, "migrations": list(controller.events),
               "requests": [{key: result[key] for key in ("request_id", "rank_initial", "output_tokens",
                                                          "token_sha256", "max_token_gap_s")} for result in results]}
    save(root / "dataplane.json", summary)
    controller.events.clear()
    controller.transactions.clear()
    return summary
