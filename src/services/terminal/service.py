import json
import os
import posixpath
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.config import DOCKER_COMPUTER_IMAGE, DOCKER_COMPUTER_MEMORY_LIMIT


class DockerTerminalService:
    def __init__(
        self,
        bot_id: str,
        runtime_dir: str,
        storage_dir: str,
        docker_image: str = DOCKER_COMPUTER_IMAGE,
        memory_limit: str = DOCKER_COMPUTER_MEMORY_LIMIT,
    ):
        self.bot_id = str(bot_id or "bot").strip() or "bot"
        self.runtime_dir = os.path.abspath(str(runtime_dir or "").strip() or ".")
        self.storage_dir = os.path.abspath(str(storage_dir or "").strip() or ".")
        self.docker_image = str(docker_image or DOCKER_COMPUTER_IMAGE).strip() or DOCKER_COMPUTER_IMAGE
        self.memory_limit = str(memory_limit or DOCKER_COMPUTER_MEMORY_LIMIT).strip() or DOCKER_COMPUTER_MEMORY_LIMIT

        slug = self.normalize_container_name(self.bot_id, fallback="bot")
        self.container_name = f"computer-{slug}"
        self.container_workspace = "/workspace"

        self.terminal_dir = os.path.join(self.runtime_dir, "terminal")
        self.workspace_dir = os.path.join(self.terminal_dir, "workspace")
        self.history_path = os.path.join(self.terminal_dir, "commands.jsonl")
        self.export_dir = os.path.join(self.terminal_dir, "exports")
        os.makedirs(self.terminal_dir, exist_ok=True)
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        self._io_lock = threading.RLock()

    @staticmethod
    def normalize_container_name(name: str, fallback: str = "bot") -> str:
        raw = str(name or "").strip().lower()
        cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "-" for ch in raw)
        cleaned = _re_sub_multi_dash(cleaned).strip("-_")
        if not cleaned:
            cleaned = str(fallback or "bot").strip().lower() or "bot"
            cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "-" for ch in cleaned)
            cleaned = _re_sub_multi_dash(cleaned).strip("-_") or "bot"
        return cleaned

    @classmethod
    def from_bot_dict(cls, bot: dict):
        bot_id = str(bot.get("id") or bot.get("name") or "bot").strip() or "bot"
        runtime_dir = str(bot.get("runtime_dir") or "").strip()
        storage_dir = str(bot.get("storage_dir") or "").strip() or "."
        docker_image = str(bot.get("docker_image") or DOCKER_COMPUTER_IMAGE).strip() or DOCKER_COMPUTER_IMAGE
        memory_limit = str(bot.get("docker_memory_limit") or DOCKER_COMPUTER_MEMORY_LIMIT).strip() or DOCKER_COMPUTER_MEMORY_LIMIT
        return cls(
            bot_id=bot_id,
            runtime_dir=runtime_dir,
            storage_dir=storage_dir,
            docker_image=docker_image,
            memory_limit=memory_limit,
        )

    def get_sandbox_status(self) -> tuple[bool, str]:
        if not _docker_available():
            return False, "Docker tidak tersedia atau tidak berjalan."
        try:
            self._ensure_container()
        except Exception as e:
            return False, f"Gagal menyiapkan container Docker: {e}"
        state = _container_state(self.container_name)
        if state == "running":
            return True, ""
        return False, f"Container '{self.container_name}' tidak dalam keadaan running (state={state})."

    def execute(
        self,
        command: str,
        timeout_sec: int = 45,
        cwd: str = "",
        source: str = "unknown",
        output_limit_chars: int = 12000,
    ) -> Dict:
        cmd = str(command or "").strip()
        if not cmd:
            return {
                "ok": False,
                "error": "Command kosong.",
                "exit_code": None,
                "output": "",
                "timed_out": False,
                "duration_ms": 0,
            }

        cmd = _normalize_noninteractive_command(cmd)
        timeout = max(1, min(120, int(timeout_sec or 45)))
        try:
            work_dir = self._resolve_cwd_strict(cwd)
        except ValueError as e:
            return {
                "ok": False,
                "error": str(e),
                "exit_code": None,
                "output": "",
                "timed_out": False,
                "duration_ms": 0,
                "cwd": "",
                "command": cmd,
            }
        script = self._wrap_command(cmd, work_dir)
        started = time.time()
        started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        run_id = str(uuid.uuid4())
        timed_out = False
        exit_code = None

        try:
            proc = subprocess.run(
                ["docker", "exec", self.container_name,
                 "bash", "-lc", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            exit_code = int(proc.returncode)
            output = _merge_output(proc.stdout, proc.stderr)
            ok = exit_code == 0
            error = ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            output = _merge_output(
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
            )
            ok = False
            error = f"Command timeout setelah {timeout} detik."
        except Exception as e:
            output = ""
            ok = False
            error = str(e)

        duration_ms = int(max(0.0, (time.time() - started) * 1000.0))
        limited_output = _truncate_text(output, max_chars=max(200, int(output_limit_chars or 12000)))
        result = {
            "id": run_id,
            "bot_id": self.bot_id,
            "started_at": started_iso,
            "duration_ms": duration_ms,
            "source": str(source or "unknown"),
            "mode": "docker",
            "container": self.container_name,
            "cwd": work_dir,
            "command": cmd,
            "ok": ok,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "error": str(error or ""),
            "output": limited_output,
        }
        self._append_history(result)
        return result

    def read_history(self, limit: int = 60) -> List[Dict]:
        max_items = max(1, min(500, int(limit or 60)))
        if not os.path.exists(self.history_path):
            return []
        try:
            with self._io_lock:
                with open(self.history_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()[-max_items:]
            rows = []
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
            rows.reverse()
            return rows
        except Exception:
            return []

    def resolve_file_for_telegram(self, file_path: str, max_bytes: int, default_cwd: str = "") -> Dict:
        raw = str(file_path or "").strip()
        if not raw:
            return {"ok": False, "error": "file_path kosong"}

        container_path = self._resolve_path_in_workspace(raw, default_cwd=default_cwd)
        if not self._is_within_workspace(container_path):
            return {
                "ok": False,
                "error": f"path harus berada di dalam {self.container_workspace}",
            }

        basename = posixpath.basename(container_path) or "file.bin"

        size_proc = subprocess.run(
            ["docker", "exec", self.container_name,
             "bash", "-c", f"test -f {_sh_quote(container_path)} && wc -c < {_sh_quote(container_path)}"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
        if size_proc.returncode != 0:
            return {"ok": False, "error": f"file tidak ditemukan di container ({container_path})"}
        try:
            size = int(str(size_proc.stdout or "").strip().splitlines()[-1].strip())
        except Exception:
            return {"ok": False, "error": "gagal membaca ukuran file di container"}
        if size <= 0:
            return {"ok": False, "error": "file kosong"}
        if size > int(max_bytes):
            return {"ok": False, "error": f"ukuran {size} bytes melebihi batas {int(max_bytes)} bytes"}

        local_name = f"{uuid.uuid4().hex[:10]}_{basename}"
        local_path = os.path.join(self.export_dir, local_name)

        cp_proc = subprocess.run(
            ["docker", "cp", f"{self.container_name}:{container_path}", local_path],
            capture_output=True, timeout=30,
        )
        if cp_proc.returncode != 0:
            detail = (cp_proc.stderr or b"").decode("utf-8", errors="replace").strip()
            return {"ok": False, "error": f"gagal menyalin file dari container: {detail or 'cp failed'}"}
        if not os.path.isfile(local_path):
            return {"ok": False, "error": "gagal menyalin file dari container"}
        return {
            "ok": True,
            "path": local_path,
            "filename": basename,
            "size": size,
            "cleanup": True,
        }

    def _ensure_container(self):
        expected_bind = os.path.normcase(os.path.normpath(self.workspace_dir))
        inspect = subprocess.run(
            ["docker", "container", "inspect", self.container_name],
            capture_output=True, text=True, timeout=10,
        )
        if inspect.returncode == 0:
            recreate = False
            try:
                payload = json.loads(inspect.stdout or "[]")
                container = payload[0] if payload else {}
                mounts = container.get("Mounts", [])
                ws_mount = next(
                    (
                        m for m in mounts
                        if str(m.get("Destination") or "") == self.container_workspace
                    ),
                    None,
                )
                src = os.path.normcase(os.path.normpath(str((ws_mount or {}).get("Source") or "")))
                mtype = str((ws_mount or {}).get("Type") or "").lower()
                current_image = _normalize_image_ref(container.get("Config", {}).get("Image", ""))
                target_image = _normalize_image_ref(self.docker_image)
                if mtype != "bind" or src != expected_bind:
                    recreate = True
                elif current_image and target_image and current_image != target_image:
                    recreate = True
            except Exception:
                recreate = True

            if recreate:
                subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, timeout=30)
            else:
                state = _container_state(self.container_name)
                if state != "running":
                    subprocess.run(
                        ["docker", "start", self.container_name],
                        capture_output=True, timeout=30,
                    )
                subprocess.run(
                    [
                        "docker", "update",
                        "--memory", self.memory_limit,
                        self.container_name,
                    ],
                    capture_output=True, timeout=30,
                )
                return

        os.makedirs(self.workspace_dir, exist_ok=True)
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--restart", "unless-stopped",
                "--memory", self.memory_limit,
                "--mount", f"type=bind,source={self.workspace_dir},target={self.container_workspace}",
                "-e", "TERM=xterm-256color",
                "-e", "COLUMNS=120",
                "-e", "LINES=40",
                "--workdir", self.container_workspace,
                self.docker_image,
                "sleep", "infinity",
            ],
            capture_output=True, timeout=120, check=True,
        )

    def _resolve_cwd(self, cwd: str) -> str:
        base = self.container_workspace
        raw = str(cwd or "").replace("\\", "/").strip()
        if not raw:
            return base
        if raw.startswith("/"):
            candidate = posixpath.normpath(raw)
        else:
            candidate = posixpath.normpath(posixpath.join(base, raw))
        if not self._is_within_workspace(candidate):
            return base
        return candidate

    def _resolve_cwd_strict(self, cwd: str) -> str:
        base = self.container_workspace
        raw = str(cwd or "").replace("\\", "/").strip()
        if not raw:
            return base
        if raw.startswith("/"):
            candidate = posixpath.normpath(raw)
        else:
            candidate = posixpath.normpath(posixpath.join(base, raw))
        if not self._is_within_workspace(candidate):
            raise ValueError(f"cwd harus berada di dalam {self.container_workspace}")
        return candidate

    def _resolve_path_in_workspace(self, path: str, default_cwd: str = "") -> str:
        raw = str(path or "").replace("\\", "/").strip()
        if not raw:
            return self.container_workspace
        if raw.startswith("/"):
            return posixpath.normpath(raw)
        base_dir = self._resolve_cwd(default_cwd)
        return posixpath.normpath(posixpath.join(base_dir, raw))

    def _is_within_workspace(self, path: str) -> bool:
        candidate = posixpath.normpath(str(path or "").strip() or "/")
        workspace = posixpath.normpath(self.container_workspace)
        try:
            return posixpath.commonpath([candidate, workspace]) == workspace
        except Exception:
            return False

    def _wrap_command(self, command: str, work_dir: str) -> str:
        workdir = str(work_dir or "").strip() or self.container_workspace
        cmd = str(command or "").strip()
        local_bin = posixpath.join(self.container_workspace, ".local/bin")
        pip_cache = posixpath.join(self.container_workspace, ".cache/pip")
        return (
            f"mkdir -p {_sh_quote(self.container_workspace)} "
            f"{_sh_quote(pip_cache)} "
            f"{_sh_quote(local_bin)}; "
            f"export HOME={_sh_quote(self.container_workspace)}; "
            "export TERM=xterm-256color; export COLUMNS=120; export LINES=40; "
            f"export PIP_CACHE_DIR={_sh_quote(pip_cache)}; "
            f"export PATH={_sh_quote(local_bin)}:\"$PATH\"; "
            f"cd {_sh_quote(workdir)} && {cmd}"
        )

    def _append_history(self, payload: Dict):
        try:
            line = json.dumps(payload, ensure_ascii=False)
            with self._io_lock:
                with open(self.history_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass


TerminalService = DockerTerminalService


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _container_state(container_name: str) -> str:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "not_found"
    except Exception:
        return "error"


def _normalize_noninteractive_command(command: str) -> str:
    cmd = str(command or "").strip()
    low = cmd.lower()
    if low == "htop" or low.startswith("htop "):
        if "-b" not in low:
            return f"{cmd} -b -n 1"
    return cmd


def _merge_output(stdout: str, stderr: str) -> str:
    out = str(stdout or "")
    err = str(stderr or "")
    if out and err:
        return f"{out.rstrip()}\n\n[stderr]\n{err.rstrip()}".strip()
    return (out or err or "").strip()


def _truncate_text(text: str, max_chars: int = 12000) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    tail = "\n\n...[output dipotong]..."
    keep = max(0, max_chars - len(tail))
    return s[:keep] + tail


def _re_sub_multi_dash(text: str) -> str:
    out = []
    prev_dash = False
    for ch in str(text or ""):
        if ch == "-":
            if prev_dash:
                continue
            prev_dash = True
            out.append(ch)
        else:
            prev_dash = False
            out.append(ch)
    return "".join(out)


def _sh_quote(text: str) -> str:
    s = str(text or "")
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _normalize_image_ref(image: str) -> str:
    raw = str(image or "").strip().lower()
    if not raw:
        return ""
    if ":" not in raw and "@sha256:" not in raw:
        return f"{raw}:latest"
    return raw
