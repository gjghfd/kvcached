import asyncio
from contextvars import ContextVar
import json
import math
import time


ACTIVE_DIAGNOSTICS = ContextVar('xplace_observation_diagnostics', default=None)
ACTIVE_COMPONENT = ContextVar('xplace_observation_component', default=None)


class ComponentDiagnostics:
    def __init__(self, *, clock_id, deadline_at, max_components=64, max_phases=16,
                 max_partial_bytes=65536, clock=None):
        if (not isinstance(clock_id, str) or not clock_id or len(clock_id) > 256
                or type(deadline_at) not in (int, float) or not math.isfinite(deadline_at)
                or any(type(value) is not int or not 0 < value <= limit for value, limit in (
                    (max_components, 64), (max_phases, 16), (max_partial_bytes, 65536)))):
            raise ValueError('Invalid bounded component diagnostics')
        self.clock_id = clock_id
        self.deadline_at = deadline_at
        self.max_components = max_components
        self.max_phases = max_phases
        self.max_partial_bytes = max_partial_bytes
        self.clock = clock or time.monotonic
        self.components = []
        self.dropped_components = 0

    def begin(self, source, identity):
        if len(self.components) >= self.max_components:
            self.dropped_components += 1
            return None
        component = Component(self, source, identity)
        self.components.append(component)
        return component

    def snapshot(self, *, include_partial_results):
        records, retained_bytes, omitted = [], 0, 0
        for component in self.components:
            record = dict(component.record, identity=dict(component.record['identity']),
                          phases=[dict(phase) for phase in component.record['phases']])
            record['result_retention'] = 'not_requested'
            if include_partial_results and component.result is not None:
                try:
                    encoded = json.dumps(component.result, allow_nan=False, separators=(',', ':'))
                    size = len(encoded.encode())
                    record['result_json_bytes'] = size
                    if retained_bytes + size <= self.max_partial_bytes:
                        record['partial_result'] = json.loads(encoded)
                        retained_bytes += size
                        record['result_retention'] = 'retained_diagnostic_only'
                    else:
                        record['result_retention'] = 'omitted_byte_limit'
                        omitted += 1
                except (TypeError, ValueError):
                    record['result_retention'] = 'omitted_invalid_json'
                    omitted += 1
            records.append(record)
        return dict(schema=1, diagnostic_only=True, prediction_input_eligible=False,
                    clock_id=self.clock_id, deadline_at=self.deadline_at,
                    max_components=self.max_components, max_phases=self.max_phases,
                    max_partial_bytes=self.max_partial_bytes, dropped_components=self.dropped_components,
                    partial_bytes_retained=retained_bytes, partial_results_omitted=omitted,
                    components=records)


class Component:
    def __init__(self, owner, source, identity):
        self.owner = owner
        self.result = None
        self.record = dict(source=source, identity=dict(identity), started_at=owner.clock(),
                           status='pending', phases=[], dropped_phases=0,
                           wire_bytes=None, wire_bytes_status='not_observed')

    def mark(self, phase, wire_bytes=None):
        if len(self.record['phases']) < self.owner.max_phases:
            self.record['phases'].append(dict(phase=phase, at=self.owner.clock()))
        else:
            self.record['dropped_phases'] += 1
        if wire_bytes is not None:
            self.record.update(wire_bytes=wire_bytes, wire_bytes_status='received_socket_payload')

    def finish(self, result=None, error=None):
        self.result = result
        finished_at = self.owner.clock()
        source_status = result.get('status') if isinstance(result, dict) else None
        error_type = type(error).__name__ if error is not None else (
            result.get('error') if isinstance(result, dict) else None)
        if isinstance(error, asyncio.CancelledError):
            status = 'cancelled'
        elif isinstance(error, asyncio.TimeoutError) or error_type == 'TimeoutError':
            status = 'timed_out'
        elif error is not None or source_status != 'observed':
            status = 'unavailable'
        else:
            status = 'completed'
        available_at = result.get('available_at') if isinstance(result, dict) else None
        received_before_deadline = (type(available_at) in (int, float) and math.isfinite(available_at)
                                    and available_at <= self.owner.deadline_at)
        self.record.update(status=status, source_status=source_status, error_type=error_type,
                           finished_at=finished_at, completed_before_deadline=finished_at <= self.owner.deadline_at,
                           received_before_deadline=received_before_deadline,
                           observed_by_deadline=(status == 'completed' and received_before_deadline
                                                 and finished_at <= self.owner.deadline_at))


def mark_phase(phase, *, wire_bytes=None):
    component = ACTIVE_COMPONENT.get()
    if component is not None:
        component.mark(phase, wire_bytes)


async def observe_component(source, identity, operation):
    diagnostics = ACTIVE_DIAGNOSTICS.get()
    if diagnostics is None:
        return await operation()
    component = diagnostics.begin(source, identity)
    token = ACTIVE_COMPONENT.set(component)
    try:
        result = await operation()
    except BaseException as error:
        if component is not None:
            component.finish(error=error)
        raise
    else:
        if component is not None:
            component.finish(result=result)
        return result
    finally:
        ACTIVE_COMPONENT.reset(token)


async def gather_drained(*operations):
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    group = asyncio.gather(*tasks)
    try:
        return await group
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(group, *tasks, return_exceptions=True)
