from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def derive_scope_key(root_key: bytes, scope_id: str, epoch: int, length: int = 32) -> bytes:
    """
    Derive a per-scope key from root_key and (scope_id, epoch).
    length=32 -> AES-256-GCM.
    """
    if len(root_key) < 32:
        raise ValueError("root_key must be at least 32 bytes (256 bits).")
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    info = f"sealed-agent-mem|scope={scope_id}|epoch={epoch}".encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
    )
    return hkdf.derive(root_key)


def _stable_json_list(values: Optional[Iterable[str]]) -> str:
    vals = [] if values is None else [str(v) for v in values if str(v)]
    return json.dumps(sorted(set(vals)), ensure_ascii=False, separators=(",", ":"))


def make_aad(
    scope_id: str,
    epoch: int,
    item_id: str,
    kind: str,
    created_ts: int,
    derived_from_item_ids: Optional[Iterable[str]] = None,
    source_scope_ids: Optional[Iterable[str]] = None,
) -> bytes:
    """
    Additional authenticated data binds ciphertext to metadata to prevent
    swapping attacks. We also bind provenance metadata so a storage attacker
    cannot silently rewrite dependency edges without invalidating the record.
    """
    derived_json = _stable_json_list(derived_from_item_ids)
    scopes_json = _stable_json_list(source_scope_ids)
    return (
        f"scope={scope_id}|epoch={epoch}|item={item_id}|kind={kind}|"
        f"ts={created_ts}|parents={derived_json}|sources={scopes_json}"
    ).encode("utf-8")


def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
    """
    AES-GCM: returns (nonce, ciphertext_with_tag).
    """
    if len(key) not in (16, 24, 32):
        raise ValueError("AESGCM key must be 16/24/32 bytes.")
    nonce = os.urandom(12)  # 96-bit nonce recommended
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return nonce, ct


def aead_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext, aad)
