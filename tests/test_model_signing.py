import os
import shutil
import zipfile

import pytest

from utils import model_signing as ms


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("AUTOML_SECRET_KEY", "test-secret")
    monkeypatch.setattr(ms, "_key", None)


def _model(tmp_path):
    folder = tmp_path / "model"
    (folder / "models" / "LightGBM").mkdir(parents=True)
    (folder / "predictor.pkl").write_bytes(b"pickle bytes")
    (folder / "models" / "LightGBM" / "model.pkl").write_bytes(b"more bytes")
    ms.sign_model_dir(str(folder))
    zip_base = tmp_path / "model"
    return shutil.make_archive(str(zip_base) + "_zip", "zip", root_dir=folder)


def test_signed_model_is_extracted(tmp_path):
    zip_path = _model(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    ms.verify_and_extract(zip_path, str(dest))
    assert (dest / "predictor.pkl").read_bytes() == b"pickle bytes"


def _rewrite(zip_path, out, change):
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            data = change(item.filename, data)
            if data is not None:
                dst.writestr(item.filename, data)
    return out


@pytest.mark.parametrize("change", [
    lambda name, data: b"evil" if name == "predictor.pkl" else data,           # tampered file
    lambda name, data: None if name == ms.SIGNATURE else data,                  # unsigned
    lambda name, data: b"0" * 64 if name == ms.SIGNATURE else data,             # forged signature
])
def test_tampered_models_are_rejected(tmp_path, change):
    zip_path = _rewrite(_model(tmp_path), str(tmp_path / "bad.zip"), change)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ms.UntrustedModel):
        ms.verify_and_extract(zip_path, str(dest))
    assert not (dest / "predictor.pkl").exists() or (dest / "predictor.pkl").read_bytes() == b"evil"


def test_extra_file_and_other_server_key_are_rejected(tmp_path, monkeypatch):
    zip_path = _model(tmp_path)
    extra = str(tmp_path / "extra.zip")
    shutil.copy(zip_path, extra)
    with zipfile.ZipFile(extra, "a") as z:
        z.writestr("../../evil.py", "print('x')")
    with pytest.raises(ms.UntrustedModel):
        ms.verify_and_extract(extra, str(tmp_path / "o1"))
    monkeypatch.setenv("AUTOML_SECRET_KEY", "another-server")
    monkeypatch.setattr(ms, "_key", None)
    with pytest.raises(ms.UntrustedModel):
        ms.verify_and_extract(zip_path, str(tmp_path / "o2"))
    with pytest.raises(ms.UntrustedModel):
        ms.verify_and_extract(__file__, str(tmp_path / "o3"))
