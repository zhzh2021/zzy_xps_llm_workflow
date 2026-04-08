"""
experiment_router.py

A function router using ChatOllama (Ollama backend).
- Actions: run | clarify | none | execute
- System instruction is persistent; per-call prompt only includes manifest + user text.
"""

import json
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from zzy_llm.Tools.utils import ExperimentSpec, build_manifest, run_experiment
from pathlib import Path
import json as _json
import sys
import os
import re


def _resolve_tools_dir() -> Path:
    """Return best-guess Tools directory even when packaged/moved."""
    base = Path(__file__).resolve().parents[1] / "Tools"
    candidates = [base]

    env_home = os.environ.get("ZZY_LLM_HOME")
    if env_home:
        env_base = Path(env_home)
        candidates.extend([
            env_base / "Tools",
            env_base / "zzy_llm" / "Tools",
        ])

    meipass = getattr(sys, "_MEIPASS", None)  # type: ignore[attr-defined]
    if meipass:
        meipass_base = Path(meipass)
        candidates.extend([
            meipass_base / "Tools",
            meipass_base / "zzy_llm" / "Tools",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base


TOOLS_DIR = _resolve_tools_dir()
tools_path = TOOLS_DIR / "XPS_reader"
if tools_path.exists() and str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

try:
    from enhanced_triage_fixed import should_route_to_mapper, EnhancedXPSDataTriage
    TRIAGE_AVAILABLE = True
except ImportError:
    TRIAGE_AVAILABLE = False
    EnhancedXPSDataTriage = None

# Import AgentState for structured memory
try:
    from Agentstate import AgentState, initialize_state, update_data_path, add_user_alert
    from triage_router import triage_decision_node
    AGENT_STATE_AVAILABLE = True
except ImportError:
    AGENT_STATE_AVAILABLE = False
    AgentState = dict  # Fallback


# Simple pattern to short-circuit "list experiments" queries
LIST_PATTERNS = re.compile(
    r"\b(what|which|list|show)\b.*\b(exp|experiment)s?\b", re.I)


# --------------------------------------------------------------------
# LLM prompt & parsing
# --------------------------------------------------------------------
SYSTEM_INSTRUCTION_BASE = """
You are a controller for local python jobs/workflows. I will send you manifests of available experiments and tools.

**XPS Data Triage Integration**:
- XPS files are automatically analyzed to detect data type (standard_spectra vs map_2d/map_hyperspectral)
- Triage routing: standard_spectra → XPS_reader workflow | map types → XPS_mapper workflow
- When user provides file_path, triage runs automatically to determine correct workflow
- You can reference triage results: data_type, confidence, nx/ny dimensions, region, energy_points

Return ONLY a single JSON object with fields:
{
  "action": "run" | "clarify" | "none" | "execute" | "triage",
  "experiment": "<exp_name or empty string>",
  "args": { "<k>": <v>, ... },
  "message": "<short user-facing explanation/answer>",
  "script": { "language": "python" | "powershell", "code": "<code to run>" }  // only when action="execute"
}

Guidelines:
- "run": Use when you are confident which experiment to execute AND you have all required args.
- "clarify": Use when the user intent maps to an experiment but required details/args are missing.
  * In "message", list EXACTLY which args are missing by name.
- "none": Provide a concise, direct answer in "message" when the user asks a general question or no experiment applies.
- "execute": Provide a small, safe, self-contained script in "script.code" with "language" set to "python" or "powershell".
- "triage": Use when user asks to analyze/detect XPS file type. Requires file_path in args.
- Never include additional keys. Never include markdown or code fences.
- If "run" but required args are missing, choose "clarify" instead.
""".strip()

SYSTEM_INSTRUCTION_WORKFLOW = """
You are a controller for local experiments.

Return ONLY a single JSON object with fields:
{
  "action": "run" | "clarify" | "none" | "execute",
  "experiment": "<exp_name or empty string>",
  "args": { "<k>": <v>, ... },
  "message": "<short user-facing explanation/answer>",
  "script": { "language": "python" | "powershell", "code": "<code to run>" }  // only when action="execute"
}

Guidelines:
- "run": Use when you are confident which experiment to execute AND you have all required args.
- "clarify": Use when the user intent maps to an experiment but required details/args are missing.
  * In "message", list EXACTLY which args are missing by name.
- "none": Provide a concise, direct answer in "message" when the user asks a general question or no experiment applies.
- "execute": Provide a small, safe, self-contained script in "script.code" with "language" set to "python" or "powershell".
- Never include additional keys. Never include markdown or code fences.
- If "run" but required args are missing, choose "clarify" instead.
""".strip()


def build_human_prompt(user_text: str) -> str:
    return (f"User request:\n{user_text}\n")


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> Dict[str, Any]:
    """
    Extract first JSON object from LLM text and validate minimal fields, including 'execute'.
    """
    m = _JSON_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No JSON object found in LLM response.")
    obj = json.loads(m.group(0))

    # required base fields
    if "action" not in obj:
        raise ValueError("Missing 'action' field.")
    action = obj["action"]
    if action not in ("run", "clarify", "none", "execute", "triage"):
        raise ValueError("Invalid 'action' value.")

    obj.setdefault("experiment", "")
    obj.setdefault("args", {})
    obj.setdefault("message", "")
    if not isinstance(obj["args"], dict):
        raise ValueError("'args' must be an object.")
    if not isinstance(obj["message"], str):
        raise ValueError("'message' must be a string.")

    # execute-specific validation
    if action == "execute":
        script = obj.get("script")
        if not isinstance(script, dict):
            raise ValueError(
                "Missing or invalid 'script' for action='execute'.")
        lang = script.get("language")
        code = script.get("code")
        if lang not in ("python", "powershell"):
            raise ValueError(
                "script.language must be 'python' or 'powershell'.")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("script.code must be a non-empty string.")

    return obj


# --------------------------------------------------------------------
# Router
# --------------------------------------------------------------------
class ExperimentRouter:
    """
    High-level helper you can call from your UI/worker.

    - Keeps a persistent SystemMessage with routing rules (SYSTEM_INSTRUCTION).
    - On decide(): builds a short HumanMessage with manifest + user text.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        temperature: float = 0.0,
        num_ctx: int = 4096,
        # set False if you want to pass manifest per-call
        embed_manifest_in_system: bool = True,
        state: AgentState = None,
    ):
        self.embed_manifest_in_system = embed_manifest_in_system
        
        # Initialize or use provided state
        if AGENT_STATE_AVAILABLE:
            self.state = state if state is not None else initialize_state()
        else:
            self.state = state if state is not None else {}
        
        # Base JSON contract + optional SciAgent tool policy and available tools
        system_content = SYSTEM_INSTRUCTION_BASE
        # Try to append SciAgent tool-calling policy from system_prompt.md
        try:
            sys_prompt_path = Path(__file__).resolve().parent / "system_prompt.md"
            sci_policy = sys_prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if sci_policy:
                system_content += "\n\nTool-Calling Policy (for reference):\n" + sci_policy
        except Exception:
            pass

        # Append available tools from Tools/_workflow.json if present
        try:
            tools_dir = TOOLS_DIR
            wf_path = tools_dir / "_workflow.json"
            tools_list = []
            if wf_path.exists():
                wf = _json.loads(wf_path.read_text(encoding="utf-8", errors="ignore"))
                for step in wf.get("steps", []):
                    t = step.get("tool")
                    if t:
                        tools_list.append({
                            "tool": t,
                            "args": step.get("args", {}),
                            "step": step.get("step", "")
                        })
            tools_list = [dict(t) for t in {tuple(sorted(d.items())): d for d in tools_list}.values()]
            if tools_list:
                system_content += "\n\nAvailable tools (from Tools/_workflow.json):\n" + _json.dumps(tools_list, indent=2)
        except Exception:
            pass
        if self.embed_manifest_in_system:
            manifest = build_manifest()
            system_content = (
                system_content
                + "\n\nAvailable experiments:\n"
                + json.dumps([asdict(m) for m in manifest], indent=2)
            )

        self.system_msg = SystemMessage(content=system_content)
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            model_kwargs={"format": "json"},
        )

    def decide(
        self,
        user_text: str,
    ) -> Dict[str, Any]:
        # Extract and store file paths from user input
        if AGENT_STATE_AVAILABLE:
            self._extract_context_from_input(user_text)
        
        if self.embed_manifest_in_system:
            prompt = build_human_prompt(user_text)
        else:
            manifest = build_manifest()
            manifest_json = json.dumps([asdict(m) for m in manifest], indent=2)
            prompt = f"Available experiments:\n{manifest_json}\n\n" + \
                build_human_prompt(user_text)
        
        # Add state context to prompt if available
        if AGENT_STATE_AVAILABLE and self.state.get('current_data_path'):
            context = f"\n\nCurrent context:\n"
            context += f"- Data path: {self.state['current_data_path']}\n"
            if self.state.get('data_type'):
                context += f"- Data type: {self.state['data_type']} (confidence: {self.state.get('triage_confidence', 0):.0%})\n"
                context += f"- Recommended workflow: {self.state.get('recommended_workflow')}\n"
            if self.state.get('user_alerts'):
                context += f"- Alerts: {len(self.state['user_alerts'])} message(s)\n"
            prompt += context

        msgs = [self.system_msg, HumanMessage(content=prompt)]

        resp = self.llm.invoke(msgs)
        decision = parse_action(resp.content)
        
        # Add user alerts to decision message if any
        if AGENT_STATE_AVAILABLE and self.state.get('user_alerts'):
            alerts_text = "\n\n".join(self.state['user_alerts'])
            current_msg = decision.get('message', '')
            decision['message'] = f"{alerts_text}\n\n{current_msg}" if current_msg else alerts_text
            # Clear alerts after showing them
            self.state['user_alerts'] = []

        # Ensure run/execute/triage decisions always surface a status string
        action = decision.get("action")
        if action in ("run", "execute", "triage"):
            msg = (decision.get("message") or "").strip()
            if not msg:
                decision["message"] = self._format_status_message(action, decision)
        
        return decision

    def _format_status_message(self, action: str, decision: Dict[str, Any]) -> str:
        if action == "run":
            exp = decision.get("experiment") or "(unknown)"
            args = decision.get("args") or {}
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items()) or "no args"
            return f"Routing decision: run '{exp}' with {args_str}."
        if action == "execute":
            script = decision.get("script") or {}
            lang = script.get("language") or "python"
            summary = decision.get("message") or script.get("code", "")[:80]
            summary = summary.strip()
            extra = f" Summary: {summary}" if summary else ""
            return f"Routing decision: execute a {lang} script.{extra}"
        if action == "triage":
            file_path = (decision.get("args") or {}).get("file_path")
            target = file_path or "provided data"
            return f"Routing decision: run data triage on {target}."
        return ""
    
    def _extract_context_from_input(self, user_text: str):
        """Extract file paths and run triage if files mentioned."""
        # Look for folder/directory mentions
        folder_patterns = [
            r'(?:folder|directory|path)\s+["\']?([^"\'\s]+)["\']?',
            r'in\s+([^\s]+(?:/|\\)[^\s]+)',
            r'(?:use|process|analyze)\s+([^\s]+(?:/|\\)[^\s]+)',
            r'(?:project\s*root|project_root)\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
        ]
        
        for pattern in folder_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                path_str = match.group(1)
                path = Path(path_str)
                
                # Check if path exists
                if path.exists():
                    if path.is_dir():
                        update_data_path(self.state, str(path))
                        self._maybe_add_folder_summary(path)
                        
                        # Find files and run triage on first one
                        files = list(path.glob('*.csv')) or list(path.glob('*.vms')) or list(path.glob('*.txt'))
                        if files:
                            self.state['current_file'] = str(files[0])
                            try:
                                self.state = triage_decision_node(self.state)
                            except Exception as e:
                                add_user_alert(self.state, f"Triage failed: {e}")
                    elif path.is_file():
                        self.state['current_file'] = str(path)
                        update_data_path(self.state, str(path.parent))
                        try:
                            self.state = triage_decision_node(self.state)
                        except Exception as e:
                            add_user_alert(self.state, f"Triage failed: {e}")
                    break
        
        # Also check for current working directory mentions
        if any(word in user_text.lower() for word in ['here', 'current', 'this folder', 'this directory']):
            cwd = Path.cwd()
            if self.state.get('current_data_path') is None:
                update_data_path(self.state, str(cwd))

    def _maybe_add_folder_summary(self, folder: Path):
        if not (AGENT_STATE_AVAILABLE and TRIAGE_AVAILABLE and EnhancedXPSDataTriage):
            return
        scan_folder = folder
        raw_sub = folder / "00_raw_data"
        if raw_sub.exists():
            scan_folder = raw_sub

        try:
            raw_files = [
                p for p in scan_folder.iterdir()
                if p.is_file() and not p.name.endswith('.py')
            ]
        except Exception:
            return
        if not raw_files:
            return

        map_files: List[str] = []
        standard_files: List[str] = []
        unknown_files: List[str] = []
        triage = EnhancedXPSDataTriage(debug=False)

        for raw_file in raw_files[:200]:
            try:
                result = triage.analyze_file_structure(raw_file)
                data_type = result['data_type'].value
                confidence = result['confidence']
                if confidence < 0.5:
                    unknown_files.append(raw_file.name)
                elif data_type in ('map_2d', 'map_hyperspectral'):
                    map_files.append(raw_file.name)
                elif data_type == 'standard_spectra':
                    standard_files.append(raw_file.name)
                else:
                    unknown_files.append(raw_file.name)
            except Exception:
                unknown_files.append(raw_file.name)

        relative = "." if scan_folder == folder else scan_folder.relative_to(folder)
        summary_lines = [
            f"🔍 Scanning {len(raw_files)} files in {relative} to determine workflow...",
            "📊 Triage Summary:",
        ]
        if map_files:
            summary_lines.append(f"🗺️ Map files: {len(map_files)}")
            for name in map_files[:3]:
                summary_lines.append(f"  • {name}")
            if len(map_files) > 3:
                summary_lines.append(f"  • ... and {len(map_files) - 3} more")
        if standard_files:
            summary_lines.append(f"📈 Standard files: {len(standard_files)}")
            for name in standard_files[:3]:
                summary_lines.append(f"  • {name}")
            if len(standard_files) > 3:
                summary_lines.append(f"  • ... and {len(standard_files) - 3} more")
        if unknown_files:
            summary_lines.append(f"❓ Unknown files: {len(unknown_files)}")
            for name in unknown_files[:3]:
                summary_lines.append(f"  • {name}")

        if map_files and standard_files:
            summary_lines.append("")
            summary_lines.append("⚠️ Mixed data types detected!")
            summary_lines.append(f"  • {len(map_files)} map files")
            summary_lines.append(f"  • {len(standard_files)} standard files")
            summary_lines.append("")
            summary_lines.append("✅ Automatic handling: Will run BOTH workflows sequentially")
            summary_lines.append("  1. Standard workflow (process spectra)")
            summary_lines.append("  2. Map workflow (process maps)")
            summary_lines.append("✅ Step 0: Triage & Quality Gate (Mandatory) completed")
            summary_lines.append("")
            summary_lines.append("🔀 Workflow routing: MIXED DATA (standard ➜ map)")
        elif map_files:
            summary_lines.append("")
            summary_lines.append("✅ Routing decision: MAP workflow")
        elif standard_files:
            summary_lines.append("")
            summary_lines.append("✅ Routing decision: STANDARD workflow")

        summary = "\n".join(summary_lines).strip()
        if summary:
            add_user_alert(self.state, summary)


# --------------------------------------------------------------------
# Optional: one-stop handler you can call from your worker/UI
# --------------------------------------------------------------------
def handle_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize how you act on the decision JSON.
    """
    action = decision["action"]

    if action == "run":
        ok, msg = run_experiment(
            decision["experiment"], decision.get("args", {}))
        return {"status": "ok" if ok else "error", "message": msg, "decision": decision}

    if action == "clarify":
        return {"status": "clarify", "message": decision.get("message", ""), "decision": decision}

    if action == "none":
        # General answer; display message directly
        return {"status": "answer", "message": decision.get("message", ""), "decision": decision}

    if action == "execute":
        script = decision.get("script", {})
        lang, code = script.get("language"), script.get("code", "")

        return {
            "status": "ready_to_execute",
            "message": decision.get("message", ""),
            "language": lang,
            "code": code,
            "decision": decision,
        }
    
    if action == "triage":
        # Run XPS data triage
        if not TRIAGE_AVAILABLE:
            return {
                "status": "error",
                "message": "Triage module not available",
                "decision": decision
            }
        
        file_path = decision.get("args", {}).get("file_path")
        if not file_path:
            return {
                "status": "error",
                "message": "Missing file_path for triage",
                "decision": decision
            }
        
        try:
            should_route, triage_result = should_route_to_mapper(Path(file_path))
            workflow = "map_workflow" if should_route else "standard_workflow"
            
            message = (
                f"Triage Result:\n"
                f"  Data Type: {triage_result['data_type'].value}\n"
                f"  Confidence: {triage_result['confidence']:.1%}\n"
                f"  Recommended: {workflow}\n"
                f"  Reason: {triage_result['reason']}"
            )
            
            return {
                "status": "triage_complete",
                "message": message,
                "triage_result": triage_result,
                "recommended_workflow": workflow,
                "decision": decision
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Triage failed: {str(e)}",
                "decision": decision
            }

    return {"status": "error", "message": "Unknown action.", "decision": decision}
