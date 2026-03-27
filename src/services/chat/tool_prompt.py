from typing import Iterable, List, Set


_TOOL_HELP = {
    "search_web": "- `search_web(query, topic, search_level, time_range, include_image)`: pakai untuk mencari informasi live atau verifikasi web. Jangan pakai untuk pengetahuan umum yang sudah kamu tahu tanpa perlu web. `include_image=True` hanya jika gambar hasil pencarian benar-benar berguna.",
    "create_schedule": "- `create_schedule(datetime_iso, context, priority)`: pakai untuk membuat reminder/jadwal baru dengan waktu yang jelas dan konteks singkat tentang apa yang harus diingatkan.",
    "list_schedules": "- `list_schedules(limit, priority, datetime_iso)`: pakai untuk melihat reminder yang sudah ada sebelum membatalkan, mengecek, atau merangkum jadwal user.",
    "cancel_schedule": "- `cancel_schedule(schedule_id)`: pakai hanya jika user minta membatalkan reminder tertentu dan kamu sudah punya `schedule_id` yang benar.",
    "save_memory": "- `save_memory(summary, m_type, priority)`: pakai untuk menyimpan hal penting jangka panjang tentang user, hubungan, preferensi, fakta, atau keputusan. Jangan simpan hal sepele atau sementara.",
    "list_memories": "- `list_memories(limit, query, m_type)`: pakai untuk mencari atau meninjau memory yang sudah ada, terutama sebelum update atau forget.",
    "forget_memory": "- `forget_memory(memory_id)`: pakai hanya jika user minta menghapus memory tertentu dan kamu sudah tahu `memory_id`-nya.",
    "update_memory": "- `update_memory(memory_id, summary, priority, m_type)`: pakai untuk memperbaiki memory yang sudah ada, bukan untuk membuat memory baru.",
    "announce_action": "- `announce_action(message)`: pakai sebelum aksi yang terasa lama agar user tahu kamu sedang melakukan sesuatu. Jangan dipakai untuk setiap langkah kecil.",
    "ai_personal_computer": "- `ai_personal_computer(command, timeout_sec, cwd)`: pakai untuk bekerja langsung di komputer pribadi AI, misalnya membaca file, menjalankan script, mengorganisir workspace, atau melakukan tugas terminal lain yang memang membantu kebutuhan user.",
    "inspect_images_from_ai_personal_computer": "- `inspect_images_from_ai_personal_computer(file_paths)`: pakai jika kamu perlu melihat gambar yang sudah ada di komputer pribadimu sendiri agar bisa menganalisis hasil visual itu pada loop berikutnya.",
    "send_files_from_ai_personal_computer": "- `send_files_from_ai_personal_computer(file_paths)`: pakai untuk mengirim file yang sudah ada di komputer pribadimu ke user Telegram setelah file itu benar-benar siap.",
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
