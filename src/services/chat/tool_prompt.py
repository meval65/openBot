from typing import Iterable, List, Set


_TOOL_HELP = {
    "search_web": "- `search_web(query, topic, search_level, time_range, include_image)`",
    "create_schedule": "- `create_schedule(datetime_iso, context, priority)`",
    "list_schedules": "- `list_schedules(limit, priority, datetime_iso)`",
    "cancel_schedule": "- `cancel_schedule(schedule_id)`",
    "save_memory": "- `save_memory(summary, m_type, priority)`",
    "list_memories": "- `list_memories(limit, query, m_type)`",
    "forget_memory": "- `forget_memory(memory_id)`",
    "update_memory": "- `update_memory(memory_id, summary, priority, m_type)`",
    "announce_action": "- `announce_action(message)`",
    "ai_personal_computer": "- `ai_personal_computer(command, timeout_sec, cwd)`",
    "inspect_images_from_ai_personal_computer": "- `inspect_images_from_ai_personal_computer(file_paths)`",
    "send_files_from_ai_personal_computer": "- `send_files_from_ai_personal_computer(file_paths)`",
}

_DEFAULT_TOOL_ORDER = [
    "search_web",
    "create_schedule",
    "list_schedules",
    "cancel_schedule",
    "save_memory",
    "list_memories",
    "forget_memory",
    "update_memory",
    "announce_action",
    "ai_personal_computer",
    "inspect_images_from_ai_personal_computer",
    "send_files_from_ai_personal_computer",
]


def _normalize_available_tools(available_tools: Iterable[str] = None) -> Set[str]:
    if not available_tools:
        return set(_DEFAULT_TOOL_ORDER)
    names = set()
    for item in available_tools:
        name = str(item or "").strip()
        if name in _TOOL_HELP:
            names.add(name)
    return names or set(_DEFAULT_TOOL_ORDER)


def build_tool_usage_directive(style: str = "default", available_tools: Iterable[str] = None) -> str:
    names = _normalize_available_tools(available_tools)
    ordered: List[str] = [name for name in _DEFAULT_TOOL_ORDER if name in names]
    if not ordered:
        return ""

    style_key = str(style or "default").strip().lower()
    if style_key not in {"default", "strict"}:
        style_key = "default"

    lines = [
        "[TOOL USAGE]",
        "Gunakan tool hanya bila memang membantu jawaban lebih akurat atau untuk menjalankan aksi yang diminta user.",
        "Jangan mengaku sudah memakai tool kalau belum.",
        "Jika tool gagal, jelaskan dengan jujur dan jangan mengarang hasil live/aksi yang tidak benar-benar terjadi.",
    ]
    if style_key == "strict":
        lines.append(
            "Untuk data live, file/workspace, reminder, memory, atau inspeksi gambar, utamakan tool yang sesuai sebelum menebak."
        )
    if "announce_action" in names:
        lines.append("Untuk aksi yang terasa lama, kirim `announce_action` singkat sebelum tool berat dijalankan.")

    lines.append("Tools yang tersedia saat ini:")
    lines.extend(_TOOL_HELP[name] for name in ordered if name in _TOOL_HELP)
    lines.append("[END TOOL USAGE]")
    return "\n\n" + "\n".join(lines)
