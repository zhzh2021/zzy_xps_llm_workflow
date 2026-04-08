# zzy_llm/llm_manager/ollama_utils.py
import subprocess


def list_models() -> list[str]:
    """
    Return installed Ollama model names (works on Windows / no --json flag).
    Parses plain-text table from `ollama list`.
    """
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, check=False
        )
        if r.returncode != 0 or not r.stdout.strip():
            return []
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if not lines:
            return []

        # drop header line if present
        if lines[0].lower().startswith("name"):
            lines = lines[1:]

        models = []
        for ln in lines:
            # first whitespace-separated token is the model name
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models
    except FileNotFoundError:
        # Ollama not installed or not in PATH
        return []
    except Exception:
        return []


def reset_session(session_id: str, model: str):
    """No-op placeholder for compatibility with UI code."""
    return
