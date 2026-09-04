#!/usr/bin/env python3
"""Web Push (RFC 8291 aes128gcm + RFC 8292 VAPID) sender and subscription store, 2026-09-04.

Deliberately dependency-free beyond `cryptography` (already installed and already a hard
dependency of the repo) and `aiohttp` (already the dashboard's HTTP stack). The obvious
alternative was `pip install pywebpush`, which was rejected because this code has to run on the
live trading server: adding a package there is an ops step (handoff + install + a restart with a
new dependency in the import path of a box that also runs trading_bot.py), whereas this module
adds nothing to install.

The tradeoff of hand-rolling crypto is that a mistake fails SILENTLY -- the push service returns
201 Created for a well-formed request whose ciphertext the browser cannot decrypt, so a bug looks
exactly like "notifications just don't show up". That is bought back by `selftest_rfc8291()`,
which reproduces the worked example in RFC 8291 section 5 byte-for-byte from its fixed keys and
salt. test/test_push_webpush.py runs it. If that passes, the content encryption is correct by
construction; only transport/VAPID can still be wrong, and those fail LOUDLY (4xx from the push
service, surfaced in the notifier log).

Subscription store: a plain JSON file (data/live/push_subscriptions.json). Subscriptions are
low-volume (one per browser/device the user installs the PWA on -- realistically <10) and the
dashboard is single-writer, so this needs no database. Endpoints that answer 404/410 are pruned
automatically: that is how the push services tell us a subscription is permanently dead (browser
uninstalled, permission revoked), and not pruning them means every later send retries garbage.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSCRIPTIONS_PATH = REPO_ROOT / "data" / "live" / "push_subscriptions.json"

# RFC 8188 record size. 4096 is what every browser push service accepts and is far larger than any
# notification payload we send, so every message is a single record (hence the 0x02 delimiter).
RECORD_SIZE = 4096
# RFC 8292 caps `exp` at 24h from now; 12h leaves room for clock skew on either side.
VAPID_TOKEN_TTL_SECONDS = 12 * 3600


# --------------------------------------------------------------------------------------
# base64url helpers (web push uses unpadded base64url everywhere, JWT included)
# --------------------------------------------------------------------------------------
def b64u_decode(value: str | bytes) -> bytes:
    if isinstance(value, str):
        value = value.encode("ascii")
    return base64.urlsafe_b64decode(value + b"=" * (-len(value) % 4))


def b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# --------------------------------------------------------------------------------------
# VAPID key handling
# --------------------------------------------------------------------------------------
def generate_vapid_keys() -> tuple[str, str]:
    """One-time setup helper -> (private_b64url, public_b64url).

    The public key is the uncompressed P-256 point (65 bytes, 0x04-prefixed) that the browser
    needs as `applicationServerKey`; the private key is the raw 32-byte scalar."""
    key = ec.generate_private_key(ec.SECP256R1())
    private_raw = key.private_numbers().private_value.to_bytes(32, "big")
    public_raw = key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return b64u_encode(private_raw), b64u_encode(public_raw)


def _load_vapid_private(private_b64: str) -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(int.from_bytes(b64u_decode(private_b64), "big"), ec.SECP256R1())


def vapid_public_key_from_private(private_b64: str) -> str:
    pub = _load_vapid_private(private_b64).public_key()
    return b64u_encode(
        pub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )


def vapid_authorization(endpoint: str, private_b64: str, subject: str,
                        *, now: int | None = None) -> str:
    """RFC 8292 section 2: `Authorization: vapid t=<JWT>, k=<public key>`.

    `aud` is the push service ORIGIN (scheme://host), not the full endpoint -- sending the full
    endpoint is a common mistake that FCM tolerates and Mozilla's autopush rejects with 401."""
    parsed = urlparse(endpoint)
    audience = f"{parsed.scheme}://{parsed.netloc}"
    issued = int(time.time()) if now is None else now
    header = b64u_encode(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode())
    claims = b64u_encode(json.dumps(
        {"aud": audience, "exp": issued + VAPID_TOKEN_TTL_SECONDS, "sub": subject},
        separators=(",", ":"),
    ).encode())
    signing_input = f"{header}.{claims}".encode("ascii")

    key = _load_vapid_private(private_b64)
    der_sig = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    # ES256 wants the raw r||s pair (64 bytes). `cryptography` only emits DER, so unpack it --
    # handing the DER blob straight to the push service is a silent 401.
    r, s = asym_utils.decode_dss_signature(der_sig)
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")

    jwt = f"{header}.{claims}.{b64u_encode(raw_sig)}"
    return f"vapid t={jwt}, k={vapid_public_key_from_private(private_b64)}"


# --------------------------------------------------------------------------------------
# RFC 8291 content encryption (aes128gcm)
# --------------------------------------------------------------------------------------
def _hkdf(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def encrypt_payload(plaintext: bytes, ua_public_b64: str, auth_secret_b64: str,
                    *, salt: bytes | None = None,
                    as_private: ec.EllipticCurvePrivateKey | None = None) -> bytes:
    """RFC 8291 section 3 -> the full aes128gcm body ready to POST.

    `salt` and `as_private` are injectable only so selftest_rfc8291() can pin the RFC's fixed
    values; production always takes the random defaults."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(16) if salt is None else salt
    as_private = ec.generate_private_key(ec.SECP256R1()) if as_private is None else as_private
    as_public = as_private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    ua_public = b64u_decode(ua_public_b64)
    auth_secret = b64u_decode(auth_secret_b64)

    shared = as_private.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), ua_public))

    # Note the ORDER in auth_info: user-agent key first, application-server key second. Swapping
    # them yields a valid-looking body the browser silently fails to decrypt.
    auth_info = b"WebPush: info\x00" + ua_public + as_public
    prk = _hkdf(auth_secret, shared, auth_info, 32)
    cek = _hkdf(salt, prk, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = _hkdf(salt, prk, b"Content-Encoding: nonce\x00", 12)

    # 0x02 is the last-record delimiter (RFC 8188 section 2). A single record always uses 0x02,
    # never 0x01 -- 0x01 tells the receiver another record follows and it will wait for one.
    ciphertext = AESGCM(cek).encrypt(nonce, plaintext + b"\x02", None)

    header = salt + struct.pack("!L", RECORD_SIZE) + bytes([len(as_public)]) + as_public
    return header + ciphertext


def selftest_rfc8291() -> None:
    """Reproduce the worked example in RFC 8291 section 5 exactly. Raises AssertionError on drift.

    This is the whole safety net for the hand-rolled encryption above -- see module docstring for
    why a silent ciphertext bug is otherwise indistinguishable from "push just doesn't work"."""
    plaintext = b"When I grow up, I want to be a watermelon"
    ua_public = "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"
    auth_secret = "BTBZMqHH6r4Tts7J_aSIgg"
    as_private_raw = b64u_decode("yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw")
    salt = b64u_decode("DGv6ra1nlYgDCS1FRnbzlw")
    expected = (
        "DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLoc"
        "InmYWAmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPTpK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLV"
        "WGNWQexSgSxsj_Qulcy4a-fN"
    )
    as_private = ec.derive_private_key(int.from_bytes(as_private_raw, "big"), ec.SECP256R1())
    got = encrypt_payload(plaintext, ua_public, auth_secret, salt=salt, as_private=as_private)
    assert b64u_encode(got) == expected, (
        f"RFC 8291 section 5 vector mismatch -- encryption is WRONG and pushes will be silently "
        f"undecryptable.\n  expected {expected}\n  got      {b64u_encode(got)}"
    )


# --------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------
class PushGone(Exception):
    """The push service says this subscription is permanently dead (404/410) -- prune it."""


async def send_push(session, subscription: dict[str, Any], payload: dict[str, Any],
                    *, private_b64: str, subject: str, ttl: int = 3600,
                    urgency: str = "normal") -> int:
    """POST one encrypted notification. Returns the HTTP status; raises PushGone on 404/410.

    `ttl` is how long the push service queues the message while the device is offline -- this is
    what makes a desktop that was asleep still receive the alert on wake, so it is deliberately
    generous (default 1h) rather than 0. `urgency` is passed through so the notifier can mark the
    low-tier digest as `low`, which lets mobile push services batch it instead of waking the radio.
    """
    endpoint = subscription["endpoint"]
    keys = subscription.get("keys") or {}
    body = encrypt_payload(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        keys["p256dh"], keys["auth"],
    )
    headers = {
        "Authorization": vapid_authorization(endpoint, private_b64, subject),
        "Content-Encoding": "aes128gcm",
        "Content-Type": "application/octet-stream",
        "TTL": str(ttl),
        "Urgency": urgency,
    }
    async with session.post(endpoint, data=body, headers=headers) as resp:
        if resp.status in (404, 410):
            raise PushGone(f"{resp.status} for {endpoint[:60]}...")
        if resp.status >= 400:
            detail = (await resp.text())[:200]
            raise RuntimeError(f"push failed {resp.status}: {detail}")
        return resp.status


# --------------------------------------------------------------------------------------
# Subscription store
# --------------------------------------------------------------------------------------
def subscription_id(subscription: dict[str, Any]) -> str:
    """Stable id for a subscription. Endpoints are long and contain the device token, so they are
    hashed rather than used raw as dict keys (keeps the store readable and the id log-safe)."""
    return hashlib.sha256(subscription["endpoint"].encode("utf-8")).hexdigest()[:16]


def load_subscriptions(path: Path = SUBSCRIPTIONS_PATH) -> dict[str, dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_subscriptions(subs: dict[str, dict[str, Any]], path: Path = SUBSCRIPTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: the notifier reads this file on its own schedule and a torn read would
    # look like "all subscriptions vanished" and silently stop every notification.
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(subs, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def add_subscription(subscription: dict[str, Any], *, label: str = "",
                     path: Path = SUBSCRIPTIONS_PATH) -> str:
    subs = load_subscriptions(path)
    sid = subscription_id(subscription)
    subs[sid] = {
        "endpoint": subscription["endpoint"],
        "keys": subscription.get("keys") or {},
        "label": label or subs.get(sid, {}).get("label", ""),
        "subscribed_utc": subs.get(sid, {}).get("subscribed_utc")
                          or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_subscriptions(subs, path)
    return sid


def remove_subscription(sid: str, path: Path = SUBSCRIPTIONS_PATH) -> bool:
    subs = load_subscriptions(path)
    if sid not in subs:
        return False
    del subs[sid]
    save_subscriptions(subs, path)
    return True


async def broadcast(payload: dict[str, Any], *, private_b64: str, subject: str,
                    ttl: int = 3600, urgency: str = "normal",
                    path: Path = SUBSCRIPTIONS_PATH) -> dict[str, int]:
    """Send to every stored subscription, pruning the ones the push service reports as gone.
    Returns {"sent": n, "pruned": n, "failed": n} -- never raises, so one dead device can never
    stop the others (or take down the notifier loop)."""
    from aiohttp import ClientSession, ClientTimeout

    subs = load_subscriptions(path)
    if not subs:
        return {"sent": 0, "pruned": 0, "failed": 0}
    sent = failed = 0
    gone: list[str] = []
    async with ClientSession(timeout=ClientTimeout(total=15)) as session:
        results = await asyncio.gather(*(
            send_push(session, sub, payload, private_b64=private_b64, subject=subject,
                      ttl=ttl, urgency=urgency)
            for sub in subs.values()
        ), return_exceptions=True)
    for sid, result in zip(subs.keys(), results):
        if isinstance(result, PushGone):
            gone.append(sid)
        elif isinstance(result, BaseException):
            failed += 1
            print(f"[push] send failed for {sid}: {result}", flush=True)
        else:
            sent += 1
    if gone:
        remaining = load_subscriptions(path)
        for sid in gone:
            remaining.pop(sid, None)
        save_subscriptions(remaining, path)
    return {"sent": sent, "pruned": len(gone), "failed": failed}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate-keys":
        priv, pub = generate_vapid_keys()
        print(f"VAPID_PRIVATE_KEY={priv}")
        print(f"VAPID_PUBLIC_KEY={pub}")
    else:
        selftest_rfc8291()
        print("RFC 8291 section 5 self-test: PASS")
