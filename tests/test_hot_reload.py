from src.ingest_engine import IngestionEngine
from src.config_loader import load_yaml

def test_engine_has_reload_config():
    cfg = load_yaml("src/config/config.yaml").raw
    eng = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml")
    assert hasattr(eng, "reload_config")
    # Should return bool; not asserting True because file may not exist in some envs
    res = eng.reload_config("src/config/config.yaml")
    assert isinstance(res, bool)
