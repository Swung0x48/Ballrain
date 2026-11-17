"""Reference-line resolution and the no-silent-fallback rule (ballrain.path, FeatureExtractor.reference_path).

r6 incident: the learner stored an absolute path_dir of its own machine; on the retail machine that directory did not
exist, load_tas_path returned None and the features quietly used the checkpoint polyline (retail cp 0.02 against
headless 0.46). Now path_dir is repo-relative, the asset's sha256 travels with the configuration, and a configuration
that needs a line it cannot load raises RouteError.
"""
import shutil

import pytest

from ballrain import features as FX
from ballrain import path as RP
from tests.test_features import make_tracker, obs_at

L01 = RP.REFPATH_DIR / "L01.npz"


def test_resolve_relative_and_refuse_foreign_absolute(tmp_path):
    assert RP.resolve_path_dir(None) == RP.REFPATH_DIR
    assert RP.resolve_path_dir("runtime/refpaths-alt") == RP.REPO_ROOT / "runtime" / "refpaths-alt"
    assert RP.resolve_path_dir(str(tmp_path)) == tmp_path
    with pytest.raises(RP.RouteError):
        RP.resolve_path_dir(str(tmp_path / "elsewhere" / "refpaths-alt"))


def test_portable_and_migrated_path_dir(tmp_path):
    assert RP.portable_path_dir(None) is None
    assert RP.portable_path_dir(str(RP.REPO_ROOT / "runtime" / "refpaths-alt")) == "runtime/refpaths-alt"
    assert RP.portable_path_dir("runtime/refpaths-alt") == "runtime/refpaths-alt"
    with pytest.raises(RP.RouteError):
        RP.portable_path_dir(str(tmp_path))  # outside the repo
    assert RP.migrate_path_dir("/home/someone/ballrain/repo/runtime/refpaths-alt") == "runtime/refpaths-alt"
    assert RP.migrate_path_dir("C:\\x\\repo\\runtime\\refpaths-alt") == "runtime/refpaths-alt"
    assert RP.migrate_path_dir("runtime/refpaths-alt") == "runtime/refpaths-alt"
    assert RP.migrate_path_dir(None) is None


def test_missing_asset_and_sha_mismatch_raise(tmp_path):
    with pytest.raises(RP.RouteError):
        RP.load_tas_path(1, None, str(tmp_path), strict=True)  # explicit path_dir without the asset
    assert RP.load_tas_path(1, None, str(tmp_path), strict=False) is None
    shutil.copyfile(L01, tmp_path / "L01.npz")
    assert RP.load_tas_path(1, None, str(tmp_path), strict=True, sha256=RP.file_sha256(L01)) is not None
    with pytest.raises(RP.RouteError):
        RP.load_tas_path(1, None, str(tmp_path), strict=True, sha256="0" * 64)


def test_level_with_an_asset_never_falls_back_to_the_polyline():
    # the hand-built level of test_features calls itself level 1, which has a TAS asset; without the synthetic-level
    # marker it is a "real" level 1 whose start / checkpoints do not match the asset: refused, not the polyline
    t = make_tracker()
    t.level.by_name.pop(RP.SYNTHETIC_LEVEL_MARKER)
    obs_at(t, (0, 2, 10))
    with pytest.raises(RP.RouteError):
        FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=0, bodies=0)).compute(t)
    # explicit polyline configurations and synthetic levels keep working
    FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=0, bodies=0, path_source="checkpoints")).compute(t)
    t2 = make_tracker()
    obs_at(t2, (0, 2, 10))
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=0, bodies=0))
    fx.compute(t2)
    assert fx.path.source == "checkpoints"


def test_foreign_absolute_path_dir_is_refused_by_the_features(tmp_path):
    t = make_tracker()
    t.level.by_name.pop(RP.SYNTHETIC_LEVEL_MARKER)
    obs_at(t, (0, 2, 10))
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=0, bodies=0,
                                              path_dir=str(tmp_path / "other-machine" / "refpaths-alt")))
    with pytest.raises(RP.RouteError):
        fx.compute(t)
