"""Deterministic read-only tool interface for static task artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import string


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolResult:
    action: str
    path: str
    output: str


class ReadOnlyToolset:
    def __init__(self, root: str | Path, max_bytes: int = 200_000, max_text_chars: int = 12_000) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.max_text_chars = max_text_chars
        if not self.root.exists() or not self.root.is_dir():
            raise ToolError(f"tool root does not exist or is not a directory: {self.root}")

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            raise ToolError("absolute paths are not allowed")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path escapes tool root") from exc
        return resolved

    def _resolve_existing_file(self, path: str | Path) -> Path:
        resolved = self._resolve(path)
        if resolved.exists():
            return resolved
        candidate = Path(path)
        if len(candidate.parts) == 1:
            matches: list[Path] = []
            for item in self.root.rglob(candidate.name):
                if not item.is_file():
                    continue
                resolved_match = item.resolve()
                try:
                    resolved_match.relative_to(self.root)
                except ValueError:
                    continue
                matches.append(resolved_match)
            matches = sorted(set(matches))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                digests = {self._sha256(item) for item in matches}
                if len(digests) == 1:
                    return min(
                        matches,
                        key=lambda item: (
                            len(item.relative_to(self.root).parts),
                            item.relative_to(self.root).as_posix(),
                        ),
                    )
                choices = ", ".join(
                    item.relative_to(self.root).as_posix() for item in matches[:10]
                )
                raise ToolError(f"ambiguous basename: {path}; candidates: {choices}")
        return resolved

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _is_probably_binary(data: bytes) -> bool:
        sample = data[:4096]
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        try:
            text = sample.decode("utf-8")
        except UnicodeDecodeError:
            allowed = set(bytes(string.printable, "ascii")) | {9, 10, 13}
            non_text_ratio = sum(byte not in allowed for byte in sample) / len(sample)
            return non_text_ratio > 0.20
        control_ratio = sum(ord(char) < 32 and char not in "\t\n\r" for char in text) / len(text)
        return control_ratio > 0.05

    def _bounded_text(self, data: bytes, path: str, total_size: int) -> str:
        sample = data[:4096]
        if self._is_probably_binary(sample):
            return f"BINARY FILE: {total_size} bytes. Use strings or hexdump on {path}."
        text = data.decode("utf-8", errors="replace")
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars] + "\n[truncated]"
        return text

    def list_files(self, path: str = ".") -> ToolResult:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"path not found: {path}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        lines = [
            str(child.relative_to(self.root))
            for child in sorted(target.rglob("*"))
            if child.is_file()
        ]
        return ToolResult("list_files", path, "\n".join(lines))

    def read_file(self, path: str) -> ToolResult:
        target = self._resolve_existing_file(path)
        if not target.exists():
            raise ToolError(f"path not found: {path}")
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        data = target.read_bytes()[: self.max_bytes]
        return ToolResult(
            "read_file",
            target.relative_to(self.root).as_posix(),
            self._bounded_text(data, path, target.stat().st_size),
        )

    def strings(self, path: str, min_length: int = 4) -> ToolResult:
        target = self._resolve_existing_file(path)
        if not target.exists() or not target.is_file():
            raise ToolError(f"file not found: {path}")
        data = target.read_bytes()[: self.max_bytes]
        printable = set(bytes(string.printable, "ascii"))
        runs: list[str] = []
        current = bytearray()
        for byte in data:
            if byte in printable and byte not in b"\x0b\x0c\r":
                current.append(byte)
            else:
                if len(current) >= min_length:
                    runs.append(current.decode("ascii", errors="ignore"))
                current.clear()
        if len(current) >= min_length:
            runs.append(current.decode("ascii", errors="ignore"))
        output = "\n".join(runs)
        if len(output) > self.max_text_chars:
            output = output[: self.max_text_chars] + "\n[truncated]"
        return ToolResult("strings", target.relative_to(self.root).as_posix(), output)

    def hexdump(self, path: str, length: int = 256) -> ToolResult:
        target = self._resolve_existing_file(path)
        if not target.exists() or not target.is_file():
            raise ToolError(f"file not found: {path}")
        data = target.read_bytes()[: min(length, self.max_bytes)]
        rows: list[str] = []
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            hex_part = " ".join(f"{byte:02x}" for byte in chunk)
            ascii_part = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in chunk)
            rows.append(f"{offset:08x}  {hex_part:<47}  |{ascii_part}|")
        return ToolResult("hexdump", target.relative_to(self.root).as_posix(), "\n".join(rows))

    def run(self, action: str, path: str = ".") -> ToolResult:
        if action == "list_files":
            return self.list_files(path)
        if action == "read_file":
            return self.read_file(path)
        if action == "strings":
            return self.strings(path)
        if action == "hexdump":
            return self.hexdump(path)
        raise ToolError(f"unsupported action: {action}")

    def execute(self, action_dict: dict[str, object]) -> dict[str, object]:
        """Execute a parsed model action and return a JSONL-safe record."""

        action = action_dict.get("action")
        path = action_dict.get("path", ".")
        if not isinstance(action, str):
            return {"ok": False, "tool": None, "path": path, "output": "", "error": "missing string action"}
        if not isinstance(path, str):
            return {"ok": False, "tool": action, "path": path, "output": "", "error": "path must be a string"}
        try:
            result = self.run(action, path)
        except ToolError as exc:
            return {"ok": False, "tool": action, "path": path, "output": "", "error": str(exc)}
        return {"ok": True, "tool": result.action, "path": result.path, "output": result.output, "error": None}


ToolSandbox = ReadOnlyToolset
