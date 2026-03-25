import inspect
import logging
from typing import Callable, Dict, List, Optional

from google.genai import types

logger = logging.getLogger(__name__)

# Calls that should not consume remote tool-call budget.
NON_BUDGETED_TOOL_NAMES = {
    "announce_action",
}


def build_tool_registry(tools: Optional[list]) -> Dict[str, Callable]:
    registry: Dict[str, Callable] = {}
    for fn in (tools or []):
        name = str(getattr(fn, "__name__", "") or "").strip()
        if not name:
            continue
        registry[name] = fn
    return registry


def sanitize_tool_kwargs(fn: Callable, raw_args: Optional[dict]) -> dict:
    args = raw_args if isinstance(raw_args, dict) else {}
    try:
        sig = inspect.signature(fn)
    except Exception:
        return args

    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return args

    allowed = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in args.items() if k in allowed}


def run_single_tool(tool_registry: dict, name: str, args: Optional[dict]) -> str:
    fn = tool_registry.get(name)
    if fn is None:
        return f"[TOOL_ERROR {name}] Tool tidak ditemukan."
    try:
        kwargs = sanitize_tool_kwargs(fn, args)
        result = fn(**kwargs)
        return str(result) if result is not None else ""
    except Exception as e:
        logger.error("[TOOL %s] Execution failed: %s", name, e)
        return f"[TOOL_ERROR {name}] {e}"


def execute_tool_calls(
    tool_registry: dict,
    function_calls: List[types.FunctionCall],
) -> List[dict]:
    prepared = []
    for fc in function_calls:
        name = str(getattr(fc, "name", "") or "").strip()
        if not name:
            continue
        args = getattr(fc, "args", None)
        prepared.append({"name": name, "args": args})

    if not prepared:
        return []

    results: List[Optional[dict]] = [None] * len(prepared)
    for idx, item in enumerate(prepared):
        output = run_single_tool(tool_registry, item["name"], item["args"])
        results[idx] = {"name": item["name"], "output": output, "parallel": False}

    return [r for r in results if isinstance(r, dict)]
