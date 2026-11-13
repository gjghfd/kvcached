import asyncio
import copy
import json
import math
from pathlib import Path
import time

from live_observation import endpoint_path, host_clock_id, read_endpoint
from observation_diagnostics import ACTIVE_DIAGNOSTICS, ComponentDiagnostics, gather_drained, mark_phase, observe_component


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class CausalCollection:
    def __init__(self, *, case_id, clock_id, period_s, deadline_s, max_events, max_bytes, max_requests):
        if (not isinstance(case_id, str) or not case_id or len(case_id) > 256
                or not isinstance(clock_id, str) or not clock_id or len(clock_id) > 256
                or not positive_number(period_s) or not positive_number(deadline_s)
                or any(type(value) is not int or value <= 0 for value in (max_events, max_bytes, max_requests))
                or max_events > 1_000_000 or max_bytes > 512 * 1024 ** 2 or max_requests > 4096):
            raise ValueError('Invalid bounded collection protocol')
        if host_clock_id() != clock_id:
            raise ValueError('Collection clock mismatch')
        self.protocol = dict(case_id=case_id, clock_id=clock_id, period_s=period_s, deadline_s=deadline_s,
                             max_events=max_events, max_bytes=max_bytes, max_requests=max_requests)
        self.events = []
        self.bytes_retained = 0
        self.event_sequence = 0
        self.dropped = 0
        self.invalid = 0
        self.requests = {}
        self.task = None
        self.running = False
        self.started = False
        self.closed = False
        self.missed_ticks = 0
        self.rounds = 0
        self.incomplete_rounds = 0
        self.sampler_error = None
        self.trace_cutoff = None

    def emit(self, kind, **payload):
        self.event_sequence += 1
        if self.closed or len(self.events) >= self.protocol['max_events']:
            self.dropped += 1
            return False
        record = dict(kind=kind, sequence=self.event_sequence, case_id=self.protocol['case_id'],
                      clock_id=self.protocol['clock_id'], recorded_at=time.monotonic(), **payload)
        try:
            encoded = json.dumps(record, allow_nan=False, separators=(',', ':'))
            size = len(encoded.encode())
        except (TypeError, ValueError):
            self.invalid += 1
            self.dropped += 1
            return False
        if len(self.events) >= self.protocol['max_events'] or self.bytes_retained + size > self.protocol['max_bytes']:
            self.dropped += 1
            return False
        self.events.append(encoded)
        self.bytes_retained += size
        return True

    def register_arrival(self, request_id, model, prompt_tokens, arrived_at):
        if (not isinstance(request_id, str) or not request_id or len(request_id) > 256
                or type(model) is not int or model not in (0, 1)
                or type(prompt_tokens) is not int or prompt_tokens <= 0
                or type(arrived_at) not in (int, float) or not math.isfinite(arrived_at)
                or not 0 <= arrived_at <= time.monotonic() or request_id in self.requests or self.closed
                or len(self.requests) >= self.protocol['max_requests']):
            self.invalid += 1
            self.emit('registration_unavailable', request_id=str(request_id)[:256])
            return False
        state = dict(request_id=request_id, model=model, prompt_tokens=prompt_tokens, arrived_at=arrived_at,
                     submitted_at=None, rank_initial=None, count=0, last_received_at=None,
                     final_output_received=False, stream_terminal=None, receipt_status='unsubmitted')
        self.requests[request_id] = state
        self.emit('arrival', **state)
        return True

    def begin_request(self, request_id, model, rank, prompt_tokens, arrived_at, submitted_at, sampling_params):
        state = self.requests.get(request_id)
        supported = (state is not None and state['model'] == model and state['prompt_tokens'] == prompt_tokens
                     and state['arrived_at'] == arrived_at and state['submitted_at'] is None
                     and getattr(getattr(sampling_params, 'output_kind', None), 'name', None) == 'CUMULATIVE'
                     and type(rank) is int and rank >= 0 and not self.closed)
        if not supported:
            self.invalid += 1
            self.emit('submission_unavailable', request_id=request_id)
            return None
        state.update(submitted_at=submitted_at, rank_initial=rank, receipt_status='observed')
        self.emit('submission', **state)
        return request_id

    def receive(self, request_id, output, received_at):
        if self.closed:
            self.emit('receipt_after_cutoff', request_id=request_id)
            return
        state = self.requests.get(request_id)
        try:
            if (state is None or state['submitted_at'] is None or state['receipt_status'] != 'observed'
                    or state['final_output_received'] or output.request_id != request_id
                    or len(output.outputs) != 1 or type(output.finished) is not bool):
                raise ValueError('Unsupported output identity or state')
            completion = output.outputs[0]
            count = len(completion.token_ids)
            previous_at = state['last_received_at'] if state['last_received_at'] is not None else state['submitted_at']
            if (completion.index != 0 or type(completion.index) is not int or count < state['count']
                    or type(received_at) not in (int, float) or not math.isfinite(received_at)
                    or not previous_at <= received_at <= time.monotonic()
                    or completion.finish_reason not in (None, 'stop', 'length', 'abort')
                    or output.finished != (completion.finish_reason is not None)
                    or type(completion.stop_reason) not in (type(None), str, int)
                    or isinstance(completion.stop_reason, str) and len(completion.stop_reason) > 256):
                raise ValueError('Unsupported cumulative receipt')
            previous_count = state['count']
            state.update(count=count, last_received_at=received_at, final_output_received=output.finished)
            self.emit('output_receipt', request_id=request_id, model=state['model'], received_at=received_at,
                      previous_count=previous_count, count=count, finished=output.finished,
                      finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                      model_output_observed_at=None, coalescing_possible=True)
        except Exception:
            self.invalid += 1
            if state is not None:
                state['receipt_status'] = 'unavailable'
            self.emit('receipt_unavailable', request_id=request_id, received_at=received_at)

    def finish_request(self, request_id, *, error=None):
        if request_id is None:
            return
        if self.closed:
            self.emit('terminal_after_cutoff', request_id=request_id)
            return
        state = self.requests[request_id]
        terminal = 'cancelled' if isinstance(error, asyncio.CancelledError) else 'error' if error is not None else 'returned'
        state['stream_terminal'] = terminal
        if error is None and not state['final_output_received']:
            state['receipt_status'] = 'unavailable'
            self.invalid += 1
        self.emit('stream_terminal', request_id=request_id, terminal=terminal, received_at=time.monotonic(),
                  count=state['count'], final_output_received=state['final_output_received'],
                  error_type=type(error).__name__ if error is not None else None)

    def cohort(self):
        return [copy.deepcopy(self.requests[request_id]) for request_id in sorted(self.requests)]

    async def collect_round(self, controller, root, tick, scheduled_at):
        started_at = time.monotonic()
        deadline_at = started_at + self.protocol['deadline_s']
        diagnostics = ComponentDiagnostics(clock_id=self.protocol['clock_id'], deadline_at=deadline_at)
        token = ACTIVE_DIAGNOSTICS.set(diagnostics)
        before = self.cohort()
        tasks = [asyncio.create_task(controller.prediction_observations(root, self.protocol['deadline_s'])),
                 asyncio.create_task(collect_batch_links(root, controller.num_ranks, controller.tensor_parallel_size,
                                                        self.protocol['deadline_s']))]
        group = asyncio.gather(*tasks)
        transport, links, error, cancelled = None, None, None, False
        try:
            transport, links = await asyncio.wait_for(group, max(0.0, deadline_at - time.monotonic()))
        except asyncio.CancelledError:
            error, cancelled = 'CancelledError', True
        except Exception as failure:
            error = type(failure).__name__
        finally:
            completed_at = time.monotonic()
            try:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(group, *tasks, return_exceptions=True)
            finally:
                ACTIVE_DIAGNOSTICS.reset(token)
        if error is None and completed_at > deadline_at:
            error = 'DeadlineExceeded'
        for index, task in enumerate(tasks):
            if not task.cancelled() and task.exception() is None:
                if index == 0:
                    transport = task.result()
                else:
                    links = task.result()
        complete = (error is None and isinstance(transport, dict) and isinstance(links, dict)
                    and all(isinstance(transport.get(key), dict) for key in ('scheduler', 'execution'))
                    and transport.get('scheduler', {}).get('status') == 'observed'
                    and transport.get('execution', {}).get('status') == 'observed' and links.get('status') == 'observed')
        self.rounds += 1
        self.incomplete_rounds += int(not complete)
        self.emit('sample', tick=tick, scheduled_at=scheduled_at, started_at=started_at, as_of=time.monotonic(),
                  cohort_at_start=before, cohort_at_completion=self.cohort(), transport=transport, batch_links=links,
                  transport_complete=complete, error_type=error, cancelled=cancelled,
                  deadline_at=deadline_at, completed_at=completed_at,
                  component_diagnostics=diagnostics.snapshot(include_partial_results=not complete),
                  prediction_status='unavailable_not_estimated')
        if cancelled:
            raise asyncio.CancelledError

    async def start(self, controller, root):
        if self.started or self.closed or host_clock_id() != self.protocol['clock_id']:
            raise ValueError('Collection already used or clock mismatch')
        self.started = self.running = True
        origin = time.monotonic()
        self.emit('collection_start', origin=origin, protocol=self.protocol)

        async def sample_loop():
            tick = 0
            try:
                while self.running:
                    scheduled = origin + tick * self.protocol['period_s']
                    await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
                    if not self.running:
                        break
                    await self.collect_round(controller, root, tick, scheduled)
                    next_tick = max(tick + 1, math.floor((time.monotonic() - origin) / self.protocol['period_s']) + 1)
                    if next_tick > tick + 1:
                        missed = next_tick - tick - 1
                        self.missed_ticks += missed
                        self.emit('missed_sample_ticks', first_tick=tick + 1, count=missed)
                    tick = next_tick
            except asyncio.CancelledError:
                raise
            except Exception as failure:
                self.sampler_error = type(failure).__name__
                self.emit('sampler_failed', error_type=self.sampler_error)
            finally:
                self.running = False

        self.task = asyncio.create_task(sample_loop())

    def mark_trace_cutoff(self, at, censored_at):
        if self.trace_cutoff is None:
            self.trace_cutoff = dict(at=at, censored_at=censored_at, cohort=self.cohort())
            self.emit('trace_cutoff', **self.trace_cutoff)

    async def stop(self, *, include_report=True):
        self.running = False
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if not self.closed:
            self.emit('collection_stop', open_requests=[request_id for request_id, state in self.requests.items()
                                                       if state['stream_terminal'] is None])
            self.closed = True
        return self.report() if include_report else None

    def report(self):
        return dict(kind='causal_collection_v1', protocol=copy.deepcopy(self.protocol),
                    events=[json.loads(event) for event in self.events], requests=self.cohort(),
                    counts=dict(attempted=self.event_sequence, retained=len(self.events), dropped=self.dropped,
                                invalid=self.invalid, bytes_retained=self.bytes_retained, rounds=self.rounds,
                                incomplete_rounds=self.incomplete_rounds, missed_ticks=self.missed_ticks),
                    stopped=self.closed, task_ended=self.task is None or self.task.done(), sampler_error=self.sampler_error,
                    trace_cutoff=copy.deepcopy(self.trace_cutoff),
                    retention_lossless=self.dropped == 0, prediction_gate_passed=False, device_activity='unknown')


async def collect_batch_links(root, num_ranks, tensor_parallel_size, timeout_s):
    from live_observation import valid_timeout

    valid_timeout(timeout_s)
    if any(type(value) is not int or value < 1 for value in (num_ranks, tensor_parallel_size)):
        raise ValueError('Invalid complete batch-link topology')
    clock_id = host_clock_id()
    directory = Path(root) / 'batch-links'

    async def operation(identity, lane):
        try:
            mark_phase('path_started')
            endpoint = endpoint_path(directory, identity['model'], identity['dp_rank'], lane)
            mark_phase('path_resolved')
            return await read_endpoint(endpoint, identity, clock_id, timeout_s)
        except (OSError, ValueError) as error:
            return dict(identity, status='unavailable', failed_at=time.monotonic(), reason=str(error),
                        error=type(error).__name__)

    async def receive(identity, lane):
        return await observe_component('batch_links', identity, lambda: operation(identity, lane))

    tasks = []
    for model in (0, 1):
        for rank in range(num_ranks):
            for lane in (None, *range(tensor_parallel_size)):
                identity = {'model': model, 'dp_rank': rank, 'role': 'engine' if lane is None else 'worker'}
                if lane is not None:
                    identity['tp_rank'] = lane
                tasks.append(receive(identity, lane))
    records = await gather_drained(*tasks)
    for record in records:
        if record.get('status') == 'observed' and record.get('kind') != 'causal_batch_link_v1':
            record.update(status='unavailable', reason='unexpected_batch_link_kind')
    return dict(kind='live_batch_link_transport_v1', clock_id=clock_id, records=records,
                status='observed' if all(record['status'] == 'observed' for record in records) else 'incomplete',
                collected_at=time.monotonic(), device_activity='unknown')
