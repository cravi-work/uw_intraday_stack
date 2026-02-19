import yaml
from src.api_catalog_loader import load_api_catalog

def test_catalog_loads(tmp_path):
    # uses the repo's catalog file if present; test only loader robustness with minimal catalog
    cat = {
        "version": 1,
        "source": "unit",
        "categories": {
            "X": {"endpoints": [{"method":"GET","path":"/api/a","operationId":"op","parameters":[{"name":"q","in":"query","required":False}]}]}
        }
    }
    p = tmp_path/"cat.yaml"
    p.write_text(yaml.safe_dump(cat), encoding="utf-8")
    reg = load_api_catalog(p)
    assert reg.has("GET","/api/a")
    reg.validate_query_params("GET","/api/a", {"q": 1})
