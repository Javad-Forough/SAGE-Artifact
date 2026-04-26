# sealing.py
from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class SealedBlob:
    nonce: bytes
    ct: bytes  # ciphertext + tag


def save_sealed_blob(path: str, blob: SealedBlob) -> None:
    """Simple binary format: nonce_len(1) | nonce | ct_len(4) | ct"""
    if len(blob.nonce) > 255:
        raise ValueError("nonce too long")
    with open(path, "wb") as f:
        f.write(bytes([len(blob.nonce)]))
        f.write(blob.nonce)
        f.write(len(blob.ct).to_bytes(4, "big"))
        f.write(blob.ct)


def load_sealed_blob(path: str) -> SealedBlob:
    with open(path, "rb") as f:
        nonce_len = int.from_bytes(f.read(1), "big")
        nonce = f.read(nonce_len)
        ct_len = int.from_bytes(f.read(4), "big")
        ct = f.read(ct_len)
    return SealedBlob(nonce=nonce, ct=ct)


class OSKeyringSealer:
    """
    Sealer backed by the OS credential store (keyring library).

    The 32-byte master key lives in the OS keyring, NOT on the filesystem.
    An attacker with read access to disk cannot unseal the root key without
    also compromising the OS credential store.

    Backend by platform:
      Linux   — libsecret / GNOME Keyring / KWallet (SecretService D-Bus)
      macOS   — macOS Keychain
      Windows — DPAPI / Windows Credential Manager

    Install: pip install keyring
    Linux headless additionally needs: pip install secretstorage
    """

    _SERVICE = "sage-memory-service"
    _ACCOUNT = "sage-master-key-v1"
    _AAD = b"sage-oskey-seal-v1"

    def __init__(self) -> None:
        import base64
        try:
            import keyring as _kr
        except ImportError as exc:
            raise RuntimeError(
                "OSKeyringSealer requires the 'keyring' package.\n"
                "Install: pip install keyring\n"
                "Linux headless: pip install keyring secretstorage\n"
                "For CI/dev without a keyring daemon use DevSealer or EnvSealer."
            ) from exc

        raw = _kr.get_password(self._SERVICE, self._ACCOUNT)
        if raw:
            self._key = base64.b64decode(raw.encode("ascii"))
        else:
            key = os.urandom(32)
            _kr.set_password(
                self._SERVICE,
                self._ACCOUNT,
                base64.b64encode(key).decode("ascii"),
            )
            self._key = key

    def seal(self, plaintext: bytes) -> SealedBlob:
        nonce = os.urandom(12)
        ct = AESGCM(self._key).encrypt(nonce, plaintext, self._AAD)
        return SealedBlob(nonce=nonce, ct=ct)

    def unseal(self, blob: SealedBlob) -> bytes:
        return AESGCM(self._key).decrypt(blob.nonce, blob.ct, self._AAD)


class EnvSealer:
    """
    Sealer whose master key is read from the SAGE_MASTER_KEY_HEX environment
    variable (64 hex chars = 32 bytes).

    Suitable for container/CI environments that inject secrets via env vars
    (Docker secrets, GitHub Actions secrets, Kubernetes secretEnv).
    The key never touches the filesystem.

    Generate a key:
        python -c "import os,binascii; print(binascii.hexlify(os.urandom(32)).decode())"
    """

    _ENV_VAR = "SAGE_MASTER_KEY_HEX"
    _AAD = b"sage-envkey-seal-v1"

    def __init__(self) -> None:
        val = os.environ.get(self._ENV_VAR, "")
        if not val or len(val) != 64:
            raise RuntimeError(
                f"EnvSealer requires {self._ENV_VAR} set to 64 hex chars (32 bytes).\n"
                "Generate: python -c \"import os,binascii; "
                "print(binascii.hexlify(os.urandom(32)).decode())\""
            )
        self._key = bytes.fromhex(val)

    def seal(self, plaintext: bytes) -> SealedBlob:
        nonce = os.urandom(12)
        ct = AESGCM(self._key).encrypt(nonce, plaintext, self._AAD)
        return SealedBlob(nonce=nonce, ct=ct)

    def unseal(self, blob: SealedBlob) -> bytes:
        return AESGCM(self._key).decrypt(blob.nonce, blob.ct, self._AAD)


class DevSealer:
    """
    DEV / TEST ONLY: master key stored in a local plaintext file.

    WARNING: anyone with filesystem read access can unseal everything.
    Use this ONLY for local development and reproducible benchmarks where
    no keyring daemon, env-secret injection, or TPM is available.

    For real deployments use OSKeyringSealer, EnvSealer, or TPMSealer.
    On Arm CCA Realms replace with realm_seal() / realm_unseal().
    """

    _AAD = b"dev-seal-v1"

    def __init__(self, master_key_path: str = "dev_master_key.bin") -> None:
        self.master_key_path = master_key_path
        self._master_key = self._load_or_create_master_key()

    def _load_or_create_master_key(self) -> bytes:
        if os.path.exists(self.master_key_path):
            with open(self.master_key_path, "rb") as f:
                k = f.read()
            if len(k) != 32:
                raise ValueError("dev master key must be exactly 32 bytes")
            return k
        k = os.urandom(32)
        with open(self.master_key_path, "wb") as f:
            f.write(k)
        return k

    def seal(self, plaintext: bytes) -> SealedBlob:
        nonce = os.urandom(12)
        ct = AESGCM(self._master_key).encrypt(nonce, plaintext, self._AAD)
        return SealedBlob(nonce=nonce, ct=ct)

    def unseal(self, blob: SealedBlob) -> bytes:
        return AESGCM(self._master_key).decrypt(blob.nonce, blob.ct, self._AAD)


def make_sealer(mode: str = "dev", **kwargs):
    """
    Factory that selects a sealer at runtime.

    mode="dev"     -> DevSealer        (local file, for experiments/CI)
    mode="env"     -> EnvSealer        (SAGE_MASTER_KEY_HEX env var)
    mode="keyring" -> OSKeyringSealer  (OS credential store)
    mode="tpm"     -> TPMSealer        (hardware TPM via tpm2-tools)
    """
    if mode == "dev":
        mp = kwargs.get("master_key_path")
        return DevSealer(master_key_path=mp) if mp else DevSealer()
    if mode == "env":
        return EnvSealer()
    if mode == "keyring":
        return OSKeyringSealer()
    if mode == "tpm":
        from tpm_sealer import TPMSealer
        return TPMSealer(**{k: v for k, v in kwargs.items() if k == "blob_prefix"})
    raise ValueError(f"Unknown sealer mode: {mode!r}. Choose: dev, env, keyring, tpm")
