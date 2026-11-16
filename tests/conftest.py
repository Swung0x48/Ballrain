import os
from pathlib import Path

import pytest

from ballrain.instances import REPO_ROOT


def tas_dir() -> Path:
    return Path(os.environ.get("BALLRAIN_TAS_DIR") or (REPO_ROOT / "runtime" / "ref" / "ModLoader" / "TASRecords"))


def tas_files():
    d = tas_dir()
    return sorted(d.glob("*.tas")) if d.is_dir() else []


@pytest.fixture
def tas_paths():
    files = tas_files()
    if not files:
        pytest.skip(f"no .tas files in {tas_dir()} (set BALLRAIN_TAS_DIR)")
    return files
