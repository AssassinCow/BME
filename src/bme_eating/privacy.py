from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path


def normalize_subject_id(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    return "".join(character for character in text if character.isalnum())


def load_or_create_subject_salt(output_root: Path) -> bytes:
    env_salt = os.environ.get("BME_SUBJECT_SALT")
    if env_salt:
        return env_salt.encode("utf-8")
    private_dir = output_root / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    salt_path = private_dir / "subject_salt.hex"
    if salt_path.exists():
        return bytes.fromhex(salt_path.read_text(encoding="ascii").strip())
    salt = secrets.token_bytes(32)
    salt_path.write_text(salt.hex(), encoding="ascii")
    return salt


def subject_key(normalized_subject_id: str, salt: bytes) -> str:
    digest = hmac.new(salt, normalized_subject_id.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:20]

