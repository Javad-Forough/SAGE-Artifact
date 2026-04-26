# tpm_sealer.py
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


class TPMSealer:
    """
    TPM-backed blob sealer built on tpm2-tools.

    Supports compact blobs such as 32-byte root keys and MAC keys.

    API expected by service_tpm.py:
      - exists(path)
      - seal_to_file(data, path)
      - unseal_from_file(path)
    """

    def __init__(self, blob_prefix: str = "results/tpm"):
        self.blob_prefix = blob_prefix
        Path(self.blob_prefix).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def artifact_paths_for(path: str) -> dict[str, str]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return {
            "pub": str(p.with_suffix(p.suffix + ".pub")),
            "priv": str(p.with_suffix(p.suffix + ".priv")),
            "primary": str(p.with_suffix(p.suffix + ".primary.ctx")),
            "ctx": str(p.with_suffix(p.suffix + ".ctx")),
            "plain": str(p),
        }

    def _run(
        self,
        args: list[str],
        *,
        input_bytes: Optional[bytes] = None,
    ) -> subprocess.CompletedProcess:
        cp = subprocess.run(
            args,
            input=input_bytes,
            capture_output=True,
        )
        if cp.returncode != 0:
            stdout = cp.stdout.decode(errors="replace") if cp.stdout else ""
            stderr = cp.stderr.decode(errors="replace") if cp.stderr else ""
            raise RuntimeError(
                f"TPM command failed: {' '.join(args)}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        return cp

    def _artifact_paths(self, path: str) -> dict[str, str]:
        return self.artifact_paths_for(path)

    @staticmethod
    def _owner_auth_args() -> list[str]:
        """
        Explicitly pass empty owner authorization.

        On the current swtpm+tpm2-tools setup, some NV commands fail when the
        owner auth is merely omitted, even though the owner hierarchy has no
        password set. Passing an explicit empty auth keeps the behavior stable.
        """
        return ["-P", ""]

    def exists(self, path: str) -> bool:
        a = self._artifact_paths(path)
        return os.path.exists(a["pub"]) and os.path.exists(a["priv"])

    def local_artifact_files(self, path: str) -> list[str]:
        a = self._artifact_paths(path)
        return [a["pub"], a["priv"], a["primary"], a["ctx"]]

    def _flush_context(self, ctx_path: str) -> None:
        try:
            self._run(["tpm2_flushcontext", ctx_path])
        except Exception:
            pass

    def _flush_all_transients(self) -> None:
        """
        Best-effort cleanup of stale transient handles left by earlier failed runs.
        """
        try:
            self._run(["tpm2_flushcontext", "-t"])
            return
        except Exception:
            pass

        try:
            cp = self._run(["tpm2_getcap", "handles-transient"])
            text = cp.stdout.decode(errors="replace").strip()
            for line in text.splitlines():
                handle = line.strip()
                if handle.startswith("0x"):
                    try:
                        self._run(["tpm2_flushcontext", handle])
                    except Exception:
                        pass
        except Exception:
            pass

    def _create_primary(self, primary_ctx: str) -> None:
        # Prevent object-context exhaustion across repeated runs.
        self._flush_all_transients()
        self._run(["tpm2_createprimary", "-C", "o", "-c", primary_ctx])

    def seal_to_file(self, data: bytes, path: str) -> None:
        a = self._artifact_paths(path)
        tmp_plain = a["plain"] + ".tmp"

        with open(tmp_plain, "wb") as f:
            f.write(data)

        try:
            self._create_primary(a["primary"])

            self._run([
                "tpm2_create",
                "-C", a["primary"],
                "-u", a["pub"],
                "-r", a["priv"],
                "-i", tmp_plain,
            ])
        finally:
            self._flush_context(a["primary"])

            for fp in (tmp_plain, a["primary"], a["ctx"]):
                try:
                    os.remove(fp)
                except FileNotFoundError:
                    pass

    def unseal_from_file(self, path: str) -> bytes:
        a = self._artifact_paths(path)
        if not self.exists(path):
            raise FileNotFoundError(f"TPM sealed blob not found for {path}")

        try:
            self._create_primary(a["primary"])

            # Load the sealed object and write a saved context file to a["ctx"].
            self._run([
                "tpm2_load",
                "-C", a["primary"],
                "-u", a["pub"],
                "-r", a["priv"],
                "-c", a["ctx"],
            ])

            # The parent is no longer needed after load.
            self._flush_context(a["primary"])

            # Flush the transient child object after load. The saved context
            # file remains on disk and can be reloaded by tpm2_unseal.
            self._flush_context(a["ctx"])

            # Best-effort cleanup of any stale transients from earlier failures.
            self._flush_all_transients()

            cp = self._run(["tpm2_unseal", "-c", a["ctx"]])
            return cp.stdout

        finally:
            self._flush_context(a["ctx"])
            self._flush_context(a["primary"])

            for key in ("primary", "ctx"):
                try:
                    os.remove(a[key])
                except FileNotFoundError:
                    pass

    def list_nv_indices(self) -> list[int]:
        cp = self._run(["tpm2_getcap", "handles-nv-index"])
        text = cp.stdout.decode(errors="replace")
        return sorted({int(m.group(0), 16) for m in re.finditer(r"0x[0-9A-Fa-f]+", text)})

    def find_free_nv_index(
        self,
        *,
        start: int = 0x01800000,
        end: int = 0x0180FFFF,
    ) -> int:
        existing = set(self.list_nv_indices())
        for index in range(start, end + 1):
            if index not in existing:
                return index
        raise RuntimeError(f"No free TPM NV indices available in range {hex(start)}..{hex(end)}")

    def define_nv_counter(self, index: int) -> None:
        self._run(
            [
                "tpm2_nvdefine",
                "-C",
                "o",
                *self._owner_auth_args(),
                "-s",
                "8",
                "-a",
                "ownerread|ownerwrite|nt=counter",
                hex(index),
            ]
        )

    def read_nv_counter(self, index: int) -> int:
        cp = self._run(
            [
                "tpm2_nvread",
                "-C",
                "o",
                *self._owner_auth_args(),
                "-s",
                "8",
                hex(index),
            ]
        )
        raw = cp.stdout or b""

        if len(raw) == 8:
            return int.from_bytes(raw, "big")

        text = raw.decode(errors="replace").strip()
        hex_match = re.search(r"0x([0-9A-Fa-f]+)", text)
        if hex_match:
            return int(hex_match.group(1), 16)

        compact = re.sub(r"[^0-9A-Fa-f]", "", text)
        if len(compact) >= 16:
            return int(compact[:16], 16)

        raise RuntimeError(
            f"Unexpected output from tpm2_nvread for index {hex(index)}: {text!r}"
        )

    def increment_nv_counter(self, index: int) -> int:
        self._run(["tpm2_nvincrement", "-C", "o", *self._owner_auth_args(), hex(index)])
        return self.read_nv_counter(index)

    def undefine_nv_index(self, index: int) -> None:
        self._run(["tpm2_nvundefine", "-C", "o", *self._owner_auth_args(), hex(index)])
