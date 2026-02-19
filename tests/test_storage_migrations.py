import duckdb
from src.storage import DbWriter

def test_predictions_start_price_migration(tmp_path):
    db_path = tmp_path / "t.duckdb"
    lock_path = tmp_path / "t.lock"
    db = DbWriter(str(db_path), str(lock_path))
    with db.writer() as con:
        db.ensure_schema(con)
        cols = [r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()]
        assert "start_price" in cols
