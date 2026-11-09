import hashlib
from io import BytesIO

import pytest

from scripts.intake_bizbench_sources import download


@pytest.mark.parametrize("kind", ["sha256", "blob_sha"])
def test_download_accepts_only_exact_frozen_source(tmp_path, monkeypatch, kind):
    body = b"[1, 2]\n"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: BytesIO(body))
    expected = (hashlib.sha256(body).hexdigest() if kind == "sha256" else
                hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest())
    result = download("https://source.example/data", tmp_path / "source", len(body),
                      **{kind: expected})
    assert result["bytes"] == len(body)
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="digest/size"):
        download("https://source.example/data", tmp_path / "wrong", len(body),
                 **{kind: "0" * len(expected)})
    with pytest.raises(ValueError, match="exceeds"):
        download("https://source.example/data", tmp_path / "oversize", len(body) - 1,
                 **{kind: expected})
