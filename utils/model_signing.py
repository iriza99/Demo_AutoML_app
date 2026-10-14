"""Only models trained by this app can be loaded for prediction.

AutoGluon saves models with pickle, and loading a pickle runs code. A model ZIP
uploaded by someone could therefore run anything on the server. Every model this
app trains gets a manifest (SHA-256 of each file) signed with a server secret;
an uploaded ZIP is extracted and loaded only if the signature and every file
check out.

Set AUTOML_SECRET_KEY to keep models valid across restarts (on Hugging Face,
as a Space secret). Without it, a random key is created in `.secret_key`, and
when that file is lost older ZIPs are rejected.
"""
import hashlib
import hmac
import json
import logging
import os
import secrets
import zipfile

MANIFEST = "automl_manifest.json"
SIGNATURE = "automl_signature"
MAX_UNCOMPRESSED = int(os.getenv("MAX_MODEL_UNZIPPED_MB", "4096")) * 1024 * 1024
_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".secret_key")
_key = None


class UntrustedModel(Exception):
    pass


def _secret():
    global _key
    if _key is None:
        env = os.getenv("AUTOML_SECRET_KEY", "").strip()
        if env:
            _key = env.encode()
        else:
            try:
                with open(_KEY_FILE, "rb") as fh:
                    _key = fh.read().strip()
            except FileNotFoundError:
                _key = secrets.token_hex(32).encode()
                with open(_KEY_FILE, "wb") as fh:
                    fh.write(_key)
                logging.warning("AUTOML_SECRET_KEY not set: created .secret_key. "
                                "Models trained before it changes will not be accepted for prediction.")
    return _key


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign(data):
    return hmac.new(_secret(), data, hashlib.sha256).hexdigest()


def sign_model_dir(folder):
    """Write a signed manifest of every file in `folder` (call before zipping it)."""
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            if rel in (MANIFEST, SIGNATURE):
                continue
            files[rel] = _sha256(path)
    manifest = json.dumps({"version": 1, "files": files}, sort_keys=True).encode()
    with open(os.path.join(folder, MANIFEST), "wb") as fh:
        fh.write(manifest)
    with open(os.path.join(folder, SIGNATURE), "w") as fh:
        fh.write(_sign(manifest))


def verify_and_extract(zip_path, dest):
    """Extract `zip_path` into `dest` only if it is a model signed by this server."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise UntrustedModel("not a zip file") from exc
    with archive:
        names = {i.filename.rstrip("/"): i for i in archive.infolist() if not i.is_dir()}
        if MANIFEST not in names or SIGNATURE not in names:
            raise UntrustedModel("no signature")
        manifest = archive.read(MANIFEST)
        signature = archive.read(SIGNATURE).decode(errors="ignore").strip()
        if not hmac.compare_digest(signature, _sign(manifest)):
            raise UntrustedModel("bad signature")
        expected = json.loads(manifest).get("files", {})
        if set(names) - {MANIFEST, SIGNATURE} != set(expected):
            raise UntrustedModel("file list does not match")
        if sum(i.file_size for i in names.values()) > MAX_UNCOMPRESSED:
            raise UntrustedModel("too large")
        root = os.path.realpath(dest)
        for name in expected:
            target = os.path.realpath(os.path.join(root, name))
            if not target.startswith(root + os.sep):
                raise UntrustedModel("unsafe path")
        archive.extractall(root, members=list(names.values()))
    for name, digest in expected.items():
        if _sha256(os.path.join(root, name)) != digest:
            raise UntrustedModel("file was modified")
