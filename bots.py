import os
import re
import subprocess
import sys
import time
import json

from src.config import DOCKER_COMPUTER_IMAGE, DOCKER_COMPUTER_MEMORY_LIMIT

_ENV_BOT_PATTERN = re.compile(r"^\.env\.([A-Za-z0-9_-]+)$")
_START_STAGGER_SECONDS = 0.4
_MONITOR_TICK_SECONDS = 1.0
_DOCKER_IMAGE_DEFAULT = DOCKER_COMPUTER_IMAGE
_DOCKER_MEMORY_LIMIT_DEFAULT = DOCKER_COMPUTER_MEMORY_LIMIT


def _parse_env_file(path: str) -> dict:
    data = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _discover_bot_envs(root_dir: str) -> list[str]:
    return sorted(
        f for f in os.listdir(root_dir)
        if _ENV_BOT_PATTERN.match(f)
    )


def _bot_id_from_env_file(env_file: str) -> str:
    match = _ENV_BOT_PATTERN.match(str(env_file or "").strip())
    if match:
        return match.group(1)
    return env_file.replace(".env.", "", 1)


def _resolve_storage_dir(storage_value: str, bot_id: str) -> str:
    raw = str(storage_value or "").strip()
    norm = os.path.normpath(raw) if raw else ""
    placeholders = {"", ".", "storage", os.path.normpath("./storage"), os.path.normpath(".\\storage")}
    if norm in placeholders:
        return os.path.join("storage", bot_id)
    return raw


def _normalize_container_name(name: str, fallback: str = "bot") -> str:
    raw = str(name or "").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "-" for ch in raw)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-_")
    if not cleaned:
        cleaned = str(fallback or "bot").strip().lower() or "bot"
    return cleaned


def _load_bot_specs(root_dir: str) -> list[dict]:
    specs = []
    for env_file in _discover_bot_envs(root_dir):
        env_path = os.path.join(root_dir, env_file)
        if not os.path.exists(env_path):
            continue

        env_map = _parse_env_file(env_path)
        token = env_map.get("TELEGRAM_BOT_TOKEN", "")
        bot_id = _bot_id_from_env_file(env_file)
        storage = _resolve_storage_dir(env_map.get("STORAGE_DIR", ""), bot_id)
        bot_name = env_map.get("BOT_INSTANCE", "").strip() or env_map.get("BOT_NAME", "").strip() or bot_id
        docker_image = _DOCKER_IMAGE_DEFAULT
        docker_memory_limit = _DOCKER_MEMORY_LIMIT_DEFAULT

        if not token:
            print(f"[!] Skip {env_file}: TELEGRAM_BOT_TOKEN is missing")
            continue

        specs.append({
            "id": bot_id,
            "env_file": env_file,
            "env_path": env_path,
            "bot_name": bot_name,
            "token": token,
            "storage_dir": storage,
            "docker_image": docker_image,
            "docker_memory_limit": docker_memory_limit,
        })

    return specs


def _filter_conflicts(specs: list[dict]) -> list[dict]:
    token_seen = {}
    storage_seen = {}
    unique_specs = []

    for spec in specs:
        token = spec["token"]
        storage = os.path.normpath(str(spec["storage_dir"] or "").strip())

        if token in token_seen:
            print(
                f"[!] Skip {spec['env_file']}: TELEGRAM_BOT_TOKEN duplicate with {token_seen[token]} "
                f"(bots must have unique token)."
            )
            continue

        if storage in storage_seen:
            print(
                f"[!] Skip {spec['env_file']}: STORAGE_DIR duplicate with {storage_seen[storage]} "
                f"(bots must have isolated storage)."
            )
            continue

        token_seen[token] = spec["env_file"]
        storage_seen[storage] = spec["env_file"]
        unique_specs.append(spec)

    return unique_specs


def _run_quick(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def _normalize_image_ref(image: str) -> str:
    raw = str(image or "").strip().lower()
    if not raw:
        return ""
    if ":" not in raw and "@sha256:" not in raw:
        return f"{raw}:latest"
    return raw


def _docker_available() -> bool:
    try:
        result = _run_quick(["docker", "info"])
        return result.returncode == 0
    except Exception:
        return False


def _ensure_docker_container(spec: dict) -> bool:
    slug = _normalize_container_name(spec.get("id", "bot"), fallback="bot")
    container_name = f"computer-{slug}"
    docker_image = str(spec.get("docker_image") or _DOCKER_IMAGE_DEFAULT).strip() or _DOCKER_IMAGE_DEFAULT
    docker_memory_limit = str(spec.get("docker_memory_limit") or _DOCKER_MEMORY_LIMIT_DEFAULT).strip() or _DOCKER_MEMORY_LIMIT_DEFAULT
    storage_dir = os.path.abspath(str(spec.get("storage_dir") or "").strip() or ".")
    host_workspace = os.path.join(storage_dir, "runtime", "terminal", "workspace")
    os.makedirs(host_workspace, exist_ok=True)
    expected_bind = os.path.normcase(os.path.normpath(host_workspace))

    inspect = _run_quick(["docker", "container", "inspect", container_name])
    if inspect.returncode == 0:
        recreate = False
        try:
            payload = json.loads(inspect.stdout or "[]")
            container = payload[0] if payload else {}
            mounts = container.get("Mounts", [])
            ws_mount = next((m for m in mounts if str(m.get("Destination") or "") == "/workspace"), None)
            src = os.path.normcase(os.path.normpath(str((ws_mount or {}).get("Source") or "")))
            mtype = str((ws_mount or {}).get("Type") or "").lower()
            current_image = _normalize_image_ref(container.get("Config", {}).get("Image", ""))
            target_image = _normalize_image_ref(docker_image)
            if mtype != "bind" or src != expected_bind:
                recreate = True
            elif current_image and target_image and current_image != target_image:
                recreate = True
        except Exception:
            recreate = True

        if recreate:
            rm = _run_quick(["docker", "rm", "-f", container_name])
            if rm.returncode != 0:
                print(f"[!] Gagal recreate container '{container_name}' untuk {spec['bot_name']}.")
                return False
        else:
            state_proc = _run_quick(["docker", "inspect", "--format", "{{.State.Status}}", container_name])
            state = state_proc.stdout.strip() if state_proc.returncode == 0 else ""
            if state != "running":
                start = _run_quick(["docker", "start", container_name])
                if start.returncode != 0:
                    print(f"[!] Gagal menghidupkan container '{container_name}' untuk {spec['bot_name']}.")
                    return False
            return True

    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "--memory", docker_memory_limit,
            "--mount", f"type=bind,source={host_workspace},target=/workspace",
            "-e", "TERM=xterm-256color",
            "-e", "COLUMNS=120",
            "-e", "LINES=40",
            "--workdir", "/workspace",
            docker_image,
            "sleep", "infinity",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        print(f"[!] Gagal membuat container '{container_name}' untuk {spec['bot_name']}: {detail}")
        return False

    print(f"[*] Container '{container_name}' bind ke '{host_workspace}' berhasil dibuat untuk {spec['bot_name']}.")
    return True


def start_bot(spec: dict, root_dir: str):
    print(f"[*] Starting bot {spec['bot_name']} using {spec['env_file']}")
    env = os.environ.copy()
    env["ENV_FILE"] = spec["env_file"]
    env["BOT_INSTANCE"] = spec["bot_name"]
    env["BOT_NAME"] = spec["bot_name"]
    env["BOT_ID"] = spec["id"]
    env["STORAGE_DIR"] = str(spec["storage_dir"])
    env["DOCKER_COMPUTER_IMAGE"] = str(spec.get("docker_image") or _DOCKER_IMAGE_DEFAULT)
    env["DOCKER_COMPUTER_MEMORY_LIMIT"] = str(spec.get("docker_memory_limit") or _DOCKER_MEMORY_LIMIT_DEFAULT)
    env["LAUNCHED_BY_BOTS_PY"] = "1"

    process = subprocess.Popen([sys.executable, "main.py"], env=env)
    return process


if __name__ == "__main__":
    print("[*] Starting AI multi-bot manager...")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    bot_specs = _load_bot_specs(root_dir)
    bot_specs = _filter_conflicts(bot_specs)

    if not bot_specs:
        print("[!] No valid bot specs found. Add files like .env.bot1 or .env.namabot.")
        sys.exit(1)

    if not _docker_available():
        print("[!] Docker tidak tersedia atau tidak berjalan. Pastikan Docker Desktop/daemon aktif.")
        sys.exit(1)

    for spec in bot_specs:
        _ensure_docker_container(spec)

    processes: list[dict] = []

    for spec in bot_specs:
        p = start_bot(spec, root_dir)
        processes.append({"name": spec["bot_name"], "spec": spec, "process": p})
        if _START_STAGGER_SECONDS > 0:
            time.sleep(_START_STAGGER_SECONDS)

    try:
        print("\n[*] All services launched. Press CTRL+C to stop all.\n")
        while True:
            time.sleep(_MONITOR_TICK_SECONDS)
            for item in processes:
                proc = item["process"]
                if proc.poll() is None:
                    continue

                name = item["name"]
                print(f"[!] {name} exited unexpectedly (code={proc.returncode}). Restarting...")
                item["process"] = start_bot(item["spec"], root_dir)

    except KeyboardInterrupt:
        print("\n[*] Stopping all AI services...")
        for item in processes:
            proc = item["process"]
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("[*] Done. All services stopped safely.")
