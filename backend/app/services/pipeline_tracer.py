"""
Pipeline Execution Tracer Service.
Tracks step-by-step latency (in milliseconds), token usage, retrieval strategy,
and intermediate chunk counts to power the frontend Execution Pipeline Tracer dashboard.
"""

import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from app.models.schemas import TraceStep, PipelineTrace


class PipelineTracer:
    """
    In-Memory Pipeline Execution Tracer:
    1. Records execution durations for every discrete step in milliseconds.
    2. Provides a python context manager (`step_timer`) for effortless block measurement.
    3. Maintains a bounded history of pipeline execution traces identified by unique `trace_id`.
    """

    # Initializes the pipeline tracer with an in-memory trace store and history cap.
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.completed_traces: Dict[str, PipelineTrace] = {}
        self._active_contexts: Dict[str, Dict[str, Any]] = {}

    # Initiates a new pipeline execution trace context and returns its unique trace identifier.
    def start_trace(self, query: str, retrieval_strategy: str) -> str:
        trace_id = f"tr_{uuid.uuid4().hex[:8]}"
        self._active_contexts[trace_id] = {
            "query": query,
            "retrieval_strategy": retrieval_strategy,
            "start_time": time.perf_counter(),
            "steps": [],
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return trace_id

    # Appends a completed step measurement with its duration and details to an active trace.
    def record_step(
        self,
        trace_id: str,
        step_name: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> TraceStep:
        step = TraceStep(
            name=step_name,
            duration_ms=round(duration_ms, 2),
            status=status,
            details=details or {},
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        if trace_id in self._active_contexts:
            self._active_contexts[trace_id]["steps"].append(step)

        return step

    # Context manager that automatically times an enclosed code block in milliseconds and logs the step.
    @contextmanager
    def step_timer(
        self,
        trace_id: str,
        step_name: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        start = time.perf_counter()
        status = "success"
        step_details = dict(details or {})
        try:
            yield step_details
        except Exception as e:
            status = "failed"
            step_details["error"] = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_step(
                trace_id=trace_id,
                step_name=step_name,
                duration_ms=elapsed_ms,
                details=step_details,
                status=status,
            )

    # Finalizes an active trace, calculates total end-to-end latency, and stores it in history.
    def end_trace(
        self,
        trace_id: str,
        total_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineTrace:
        if trace_id not in self._active_contexts:
            raise KeyError(f"Active trace with ID '{trace_id}' not found.")

        ctx = self._active_contexts.pop(trace_id)
        total_latency = (time.perf_counter() - ctx["start_time"]) * 1000.0

        trace = PipelineTrace(
            trace_id=trace_id,
            query=ctx["query"],
            retrieval_strategy=ctx["retrieval_strategy"],
            steps=ctx["steps"],
            total_latency_ms=round(total_latency, 2),
            total_tokens=total_tokens,
            created_at=ctx["created_at"],
        )

        # Enforce maximum history capacity (FIFO eviction)
        if len(self.completed_traces) >= self.max_history:
            oldest_key = next(iter(self.completed_traces))
            del self.completed_traces[oldest_key]

        self.completed_traces[trace_id] = trace
        return trace

    def save_answer(self, trace_id, response):
        if trace_id in self.completed_traces:
            self.completed_traces[trace_id].saved_answer = response

    # Retrieves a completed pipeline trace by its unique trace identifier.
    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        return self.completed_traces.get(trace_id)

    # Returns the list of all completed pipeline traces ordered from most recent to oldest.
    def list_traces(self, limit: int = 20) -> List[PipelineTrace]:
        traces = list(self.completed_traces.values())
        return traces[::-1][:limit]

    # Clears all completed and active execution traces from memory.
    def clear(self) -> None:
        self.completed_traces.clear()
        self._active_contexts.clear()


# Global singleton instance of PipelineTracer for application-wide tracing.
class AccountTracer:
    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._tracers = {}
    def __getattr__(self, name):
        from app.core.account_context import account_id
        from app.auth import ROOT
        import json
        uid = account_id.get()
        if uid is None:
            raise RuntimeError('Account context required')
        with self._lock:
            path = ROOT / uid / 'history.json'
            if uid not in self._tracers:
                tracer = PipelineTracer(max_history=1000)
                if path.exists():
                    tracer.completed_traces = {t['trace_id']: PipelineTrace.model_validate(t) for t in json.loads(path.read_text())}
                self._tracers[uid] = tracer
            tracer = self._tracers[uid]
        method = getattr(tracer, name)
        if name not in ('end_trace', 'clear', 'save_answer'):
            return method
        def persist(*args, **kwargs):
            with self._lock:
                result = method(*args, **kwargs)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.new')
                temp.write_text(json.dumps([t.model_dump(mode='json') for t in tracer.completed_traces.values()]))
                temp.replace(path)
                return result
        return persist
pipeline_tracer = AccountTracer()

