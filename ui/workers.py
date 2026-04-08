import sys
import subprocess
import tempfile
import traceback
from typing import Dict, Any

from PySide6.QtCore import QObject, Signal

from zzy_llm.llm_manager.experiment_router import ExperimentRouter
from zzy_llm.Tools.utils import run_experiment

# Import AgentState for persistent memory
try:
    from zzy_llm.llm_manager.Agentstate import AgentState, initialize_state
    AGENT_STATE_AVAILABLE = True
except ImportError:
    AGENT_STATE_AVAILABLE = False
    AgentState = dict

# Optional triage summary formatter
try:
    from zzy_llm.llm_manager.triage_router import format_triage_summary as _format_triage_summary
    TRIAGE_SUMMARY_AVAILABLE = True
except Exception:
    TRIAGE_SUMMARY_AVAILABLE = False
    _format_triage_summary = None


class ChatWorker(QObject):
    """
    Runs the routing call in a background thread.
    Emits:
      - finished(response_text, error_text)
      - propose({"experiment", "args", "message}) for action='run'
      - propose_execute({"message","script","ui_confirm"}) for action='execute'
      - triage_summary(str) whenever new triage insights are available
    """
    finished = Signal(object, object)
    propose = Signal(object)
    propose_execute = Signal(object)
    triage_summary = Signal(str)
    status = Signal(str)

    # Class-level state shared across all chat workers
    _shared_state = None

    def __init__(self, router_model_name: str, prompt: str, max_tokens: int):
        super().__init__()
        self.router_model_name = router_model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        
        # Initialize shared state once
        if ChatWorker._shared_state is None and AGENT_STATE_AVAILABLE:
            ChatWorker._shared_state = initialize_state()
        
        # Pass state to router for persistence
        self.router = ExperimentRouter(
            model=self.router_model_name,
            state=ChatWorker._shared_state
        )

    def run(self):
        try:
            action = self.router.decide(user_text=self.prompt)
            kind = action.get("action")
            self._emit_status_if_needed(kind, action)
            self._maybe_emit_triage_summary()

            if kind == "run":
                self.propose.emit({
                    "experiment": action.get("experiment") or "",
                    "args": action.get("args") or {},
                    "message": action.get("message") or ""
                })
                self.finished.emit("", "")
                return

            if kind == "execute":
                self.propose_execute.emit({
                    "message": action.get("message") or "",
                    "script": action.get("script") or {},
                    "ui_confirm": action.get("ui_confirm") or {}
                })
                self.finished.emit("", "")
                return

            if kind == "triage":
                summary = action.get("message") or self._build_triage_summary()
                if summary:
                    self.finished.emit(summary, "")
                else:
                    self.finished.emit("Triage completed.", "")
                return

            if kind in ("clarify", "none"):
                msg = action.get("message") or "No suitable experiment. Please specify."
                self.finished.emit(msg, "")
                return

            self.finished.emit("I didn’t understand that request.", "")

        except Exception as e:
            traceback.print_exc()
            friendly = (
                "Something went wrong while contacting the LLM.\n"
                "Please make sure Ollama is running and the selected model is available."
            )
            self.finished.emit("", friendly)

    def _emit_status_if_needed(self, kind: str, action: Dict[str, Any]):
        if kind not in ("run", "execute", "triage"):
            return
        msg = (action.get("message") or "").strip()
        if msg:
            self.status.emit(msg)

    def _maybe_emit_triage_summary(self):
        summary = self._build_triage_summary()
        if summary:
            self.triage_summary.emit(summary)

    def _build_triage_summary(self) -> str:
        if not AGENT_STATE_AVAILABLE:
            return ""
        state = getattr(self.router, "state", None) or {}
        data_type = state.get("data_type")
        if not data_type:
            return ""
        signature = (
            data_type,
            state.get("triage_confidence"),
            state.get("triage_reason"),
            state.get("recommended_workflow"),
        )
        if state.get("_triage_signature") == signature:
            return ""
        state["_triage_signature"] = signature

        if TRIAGE_SUMMARY_AVAILABLE and _format_triage_summary:
            try:
                return _format_triage_summary(state)
            except Exception:
                pass

        lines = [
            "=== XPS Triage Summary ===",
            f"Data Type: {data_type}",
        ]
        conf = state.get("triage_confidence")
        if conf is not None:
            lines.append(f"Confidence: {conf:.0%}")
        workflow = state.get("recommended_workflow")
        if workflow:
            lines.append(f"Routing: {workflow}")
        reason = state.get("triage_reason")
        if reason:
            lines.append(f"Reason: {reason}")
        params = state.get("triage_parameters") or {}
        if params:
            if params.get("nx") and params.get("ny"):
                lines.append(f"Spatial: {params['nx']} x {params['ny']} = {params.get('total_pixels', '?')} px")
            if params.get("region"):
                lines.append(f"Region: {params['region']}")
            if params.get("energy_points"):
                lines.append(f"Energy Points: {params['energy_points']}")
        return "\n".join(lines)


class ExperimentExecWorker(QObject):
    """Runs a selected experiment with args in a background thread."""
    done = Signal(object, object)  # (message, error)

    def __init__(self, exp_name: str, args: Dict[str, Any]):
        super().__init__()
        self.exp_name = exp_name
        self.args = args

    def run(self):
        try:
            ok, result = run_experiment(self.exp_name, self.args)
            if ok:
                self.done.emit(result, "")
            else:
                self.done.emit("", result)
        except Exception as e:
            traceback.print_exc()
            self.done.emit("", f"Experiment error: {e}")


class ScriptExecWorker(QObject):
    """Executes a Python or PowerShell script in a background thread."""
    done = Signal(object, object)  # (message, error)

    def __init__(self, language: str, code: str):
        super().__init__()
        self.language = language
        self.code = code

    def run(self):
        try:
            if self.language == "python":
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(self.code)
                    path = f.name
                proc = subprocess.run([sys.executable, path], capture_output=True, text=True)

            elif self.language == "powershell":
                with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
                    f.write(self.code)
                    path = f.name
                # Use Windows PowerShell; switch to "pwsh" for PowerShell 7 if preferred
                proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
                                      capture_output=True, text=True)
            else:
                self.done.emit("", f"Unsupported script language: {self.language}")
                return

            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode == 0:
                msg = "Script completed.\n" + (out if out else "[no output]")
                self.done.emit(msg, "")
            else:
                self.done.emit("", f"Script failed (exit {proc.returncode}).\n{err if err else '[no stderr]'}")

        except Exception as e:
            traceback.print_exc()
            self.done.emit("", f"Script execution error: {e}")
