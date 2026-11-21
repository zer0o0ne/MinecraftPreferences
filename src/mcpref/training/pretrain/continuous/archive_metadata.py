"""Disk-backed endpoint membership, pending degrees and consumed edge IDs."""

import sqlite3
from collections.abc import Mapping
from collections.abc import Set as AbstractSet


class Keys(AbstractSet):
    def __init__(self, metadata, table):
        self.metadata, self.table = metadata, table

    def __contains__(self, key):
        return (
            self.metadata.db.execute(
                f"SELECT 1 FROM {self.table} WHERE key=?", (key,)
            ).fetchone()
            is not None
        )

    def __len__(self):
        return self.metadata.db.execute(
            f"SELECT count(*) FROM {self.table}"
        ).fetchone()[0]

    def __iter__(self):
        for (key,) in self.metadata.db.execute(
            f"SELECT key FROM {self.table} ORDER BY key"
        ):
            yield key


class Live(Keys, Mapping):
    def __getitem__(self, key):
        row = self.metadata.db.execute(
            "SELECT shard,remaining FROM live WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return dict(zip(("shard", "remaining"), row))


class Metadata:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS live (key TEXT PRIMARY KEY, shard INTEGER, remaining INTEGER);
            CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS consumed (key TEXT PRIMARY KEY);
        """)

    def put(self, key, shard, remaining):
        if remaining < 0:
            raise ValueError("Negative pending preference edge count")
        self.db.execute("INSERT INTO seen VALUES (?)", (key,))
        if remaining:
            self.db.execute("INSERT INTO live VALUES (?,?,?)", (key, shard, remaining))

    def consume(self, edge, other):
        row = self.db.execute(
            "SELECT remaining FROM live WHERE key=?", (other,)
        ).fetchone()
        if row is None or row[0] <= 0:
            raise RuntimeError("Preference edge references a retired endpoint")
        try:
            self.db.execute("INSERT INTO consumed VALUES (?)", (edge,))
        except sqlite3.IntegrityError:
            raise RuntimeError(f"preference edge {edge} consumed twice") from None
        remaining = row[0] - 1
        if remaining:
            self.db.execute(
                "UPDATE live SET remaining=? WHERE key=?", (remaining, other)
            )
        else:
            self.db.execute("DELETE FROM live WHERE key=?", (other,))
        return remaining

    def snapshot(self, path):
        self.db.commit()
        target = sqlite3.connect(path)
        try:
            self.db.backup(target)
        finally:
            target.close()

    def restore(self, path=None):
        self.db.rollback()
        if path is None:
            for table in ("live", "seen", "consumed"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.commit()
        else:
            source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                source.backup(self.db)
            finally:
                source.close()

    def close(self):
        self.db.close()
