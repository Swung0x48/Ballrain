"""f7 reach fields as task assets, server side (ballrain/pool/reach_assets.py, docs/f7-native-port.md 8): the gateway
finds every geometry of the start record's level stream (L2: two fields), ships them zstd-compressed and sha-addressed,
and the verifier replays only from byte-identical files. Uses the real SR02 record (BALLRAIN_RUNTIME's dataset) and a
temporary copy of the field cache (never writes the live <runtime>/fastreach)."""
import hashlib
import shutil
import sys
from pathlib import Path

import pytest

from ballrain import instances
from ballrain.pool import reach_assets as RA

REC = instances.runtime_dir() / "datasets" / "tas-v1" / "SR02_1_28_992_264"


@pytest.fixture(scope="module")
def fields(tmp_path_factory):
    if not (REC / "level.msg").is_file():
        pytest.skip("no SR02 record on this machine")
    from ballrain import fastreach as FR
    from ballrain.reach_features import reach_version
    cache = tmp_path_factory.mktemp("fastreach")
    live = instances.runtime_dir() / "fastreach"
    for f in live.glob("L02-*.npz"):  # a copy of what is already built (else the test builds: numba, ~4 s each)
        shutil.copy2(f, cache / f.name)
    if not FR.HAVE_NUMBA and not list(cache.glob("L02-*.npz")):
        pytest.skip("no numba and no built L02 fields")
    assets = tmp_path_factory.mktemp("assets")
    entries = RA.export_reach_assets(REC, 2, reach_version(), assets, cache=cache)
    return cache, assets, entries


def test_l2_record_ships_both_geometries(fields):
    cache, assets, entries = fields
    assert len(entries) == 2 and len({e["key_sha256"] for e in entries}) == 2
    for e in entries:
        f = cache / RA.field_name(2, e["key_sha256"])
        assert hashlib.sha256(f.read_bytes()).hexdigest() == e["sha256"] and f.stat().st_size == e["size"]
        z = assets / f"{e['zsha256']}.zst"
        assert z.stat().st_size == e["zsize"] < e["size"] // 8 and e["zsize"] <= RA.MAX_ZSIZE
        assert set(e) == {"level", "key_sha256", "sha256", "size", "zsha256", "zsize"}
    # the same export again: the same entries (fields belong to the geometry, not to a policy)
    from ballrain.reach_features import reach_version
    assert RA.export_reach_assets(REC, 2, reach_version(), assets, cache=cache) == entries


def test_verifier_cache_check_restores_and_detects(fields, tmp_path):
    cache, assets, entries = fields
    assert RA.check_cache(entries, cache=cache) is None
    empty = tmp_path / "cache"
    assert "missing" in RA.check_cache(entries, cache=empty)  # no asset store: replay_failed
    assert RA.check_cache(entries, assets_dir=assets, cache=empty) is None  # restored from the asset store
    assert sorted(p.name for p in empty.glob("*.npz")) == sorted(RA.field_name(2, e["key_sha256"]) for e in entries)
    bad = [dict(entries[0], sha256="0" * 64)] + entries[1:]
    assert "differs" in RA.check_cache(bad, cache=empty)
    with pytest.raises(ValueError):
        RA.install_from_assets(dict(entries[0], zsha256=entries[1]["zsha256"]), assets, tmp_path / "c2")


def test_export_refuses_another_reach_version(fields, tmp_path):
    cache, _, _ = fields
    with pytest.raises(ValueError, match="reach_version"):
        RA.export_reach_assets(REC, 2, "fastreach4.1-block1", tmp_path, cache=cache)


def test_gateway_serves_zst_assets_only_from_the_asset_store(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_pool_gateway import Gw
    g = Gw(tmp_path)
    try:
        sha = "a" * 64
        (g.cfg.pool_dir / "assets").mkdir(parents=True, exist_ok=True)
        (g.cfg.pool_dir / "assets" / f"{sha}.zst").write_bytes(b"zstd bytes")
        r = g.call("GET", f"/pool/assets/{sha}.zst")
        assert r.status == 200 and "immutable" in dict(r.headers)["Cache-Control"]
        assert g.call("GET", f"/pool/weights/{sha}.zst").status == 404
        assert g.call("GET", f"/pool/assets/{'b' * 64}.zst").status == 404
    finally:
        g.close()
