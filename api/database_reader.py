"""
Read-only SQLite Database Layer for Stash Sense Sidecar

This is a read-only subset of the full database.py from stash-sense-trainer.
The sidecar only needs to read the performer database, not write to it.

Database is produced by stash-sense-trainer and consumed read-only here.

NOTE: If you need write operations, use the full database.py from trainer.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator


@dataclass
class Performer:
    """Performer record from the database."""
    id: int
    canonical_name: str
    disambiguation: Optional[str]
    gender: Optional[str]
    country: Optional[str]
    ethnicity: Optional[str]
    birth_date: Optional[str]
    death_date: Optional[str]
    height_cm: Optional[int]
    eye_color: Optional[str]
    hair_color: Optional[str]
    career_start_year: Optional[int]
    career_end_year: Optional[int]
    scene_count: Optional[int]
    stashdb_updated_at: Optional[str]
    face_count: int
    image_url: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class StashboxId:
    """Stash-box ID mapping."""
    performer_id: int
    endpoint: str
    stashbox_performer_id: str
    created_at: Optional[str] = None


@dataclass
class Alias:
    """Performer alias/stage name."""
    id: int
    performer_id: int
    alias: str
    source_endpoint: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Face:
    """Face embedding metadata."""
    id: int
    performer_id: int
    embedding_index: int
    image_url: str
    source_endpoint: str
    quality_score: Optional[float]
    created_at: str
    pruned: Optional[bool] = None
    yaw: Optional[float] = None


class PerformerDatabaseReader:
    """
    Read-only SQLite database for performer metadata.

    This class provides read-only access to the performer database
    produced by stash-sense-trainer.

    Usage:
        db = PerformerDatabaseReader("performers.db")

        # Get a performer
        performer = db.get_performer(123)

        # Find by stash-box ID
        performer = db.get_performer_by_stashbox_id("stashdb", "abc-123-uuid")

        # Get performer by face index
        performer = db.get_performer_by_face_index(42)
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get a read-only database connection."""
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Performer Reads ====================

    def get_performer(self, performer_id: int) -> Optional[Performer]:
        """Get a performer by internal ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM performers WHERE id = ?", (performer_id,)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def get_performer_by_stashbox_id(
        self, endpoint: str, stashbox_id: str
    ) -> Optional[Performer]:
        """Get a performer by their stash-box ID."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM performers p
                JOIN stashbox_ids s ON p.id = s.performer_id
                WHERE s.endpoint = ? AND s.stashbox_performer_id = ?
                """,
                (endpoint, stashbox_id)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def get_stashbox_ids(self, performer_id: int) -> list[StashboxId]:
        """Get all stash-box IDs for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stashbox_ids WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [StashboxId(**dict(row)) for row in rows]

    def get_aliases(self, performer_id: int) -> list[Alias]:
        """Get all aliases for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM aliases WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [Alias(**dict(row)) for row in rows]

    def find_by_alias(self, alias: str) -> list[Performer]:
        """Find performers by alias (case-insensitive)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM performers p
                JOIN aliases a ON p.id = a.performer_id
                WHERE a.alias = ? COLLATE NOCASE
                """,
                (alias,)
            ).fetchall()
            return [Performer(**dict(row)) for row in rows]

    # ==================== Face Reads ====================

    def get_faces(self, performer_id: int) -> list[Face]:
        """Get all faces for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [Face(**dict(row)) for row in rows]

    def get_performer_by_face_index(
        self, embedding_index: int
    ) -> Optional[Performer]:
        """Get performer by usearch face index."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM performers p
                JOIN faces f ON p.id = f.performer_id
                WHERE f.embedding_index = ?
                """,
                (embedding_index,)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def get_max_face_index(self) -> Optional[int]:
        """Get the maximum face index in the database."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT MAX(embedding_index) FROM faces"
            ).fetchone()
            return row[0] if row and row[0] is not None else None

    # ==================== Benchmark Support ====================

    def _parse_universal_id(self, universal_id: str) -> tuple[str, str]:
        """Parse universal_id into (endpoint, stashbox_id).

        Args:
            universal_id: Format "stashdb.org:uuid-here"

        Returns:
            Tuple of (endpoint, stashbox_id)
        """
        parts = universal_id.split(":", 1)
        if len(parts) != 2:
            return ("", "")
        return (parts[0], parts[1])

    def get_face_count_for_performer(self, universal_id: str) -> int:
        """Get the face count for a performer by universal_id.

        Args:
            universal_id: Format "stashdb.org:uuid-here"

        Returns:
            The performer's face_count, or 0 if not found
        """
        endpoint, stashbox_id = self._parse_universal_id(universal_id)
        if not endpoint or not stashbox_id:
            return 0

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.face_count FROM performers p
                JOIN stashbox_ids s ON p.id = s.performer_id
                WHERE s.endpoint = ? AND s.stashbox_performer_id = ?
                """,
                (endpoint, stashbox_id)
            ).fetchone()
            if row and row[0] is not None:
                return row[0]
        return 0

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            stats['performer_count'] = conn.execute(
                "SELECT COUNT(*) FROM performers"
            ).fetchone()[0]
            stats['performers_with_faces'] = conn.execute(
                "SELECT COUNT(*) FROM performers WHERE face_count > 0"
            ).fetchone()[0]
            stats['total_faces'] = conn.execute(
                "SELECT COUNT(*) FROM faces"
            ).fetchone()[0]
            stats['total_aliases'] = conn.execute(
                "SELECT COUNT(*) FROM aliases"
            ).fetchone()[0]
            stats['stashbox_counts'] = {}
            for row in conn.execute(
                "SELECT endpoint, COUNT(*) FROM stashbox_ids GROUP BY endpoint"
            ):
                stats['stashbox_counts'][row[0]] = row[1]
            return stats

    # ==================== Iteration ====================

    def iter_performers(self, batch_size: int = 1000) -> Iterator[Performer]:
        """Iterate over all performers in batches."""
        with self._connection() as conn:
            offset = 0
            while True:
                rows = conn.execute(
                    "SELECT * FROM performers ORDER BY id LIMIT ? OFFSET ?",
                    (batch_size, offset)
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    yield Performer(**dict(row))
                offset += batch_size

    def iter_performers_with_faces(self, batch_size: int = 1000) -> Iterator[Performer]:
        """Iterate over performers that have face embeddings."""
        with self._connection() as conn:
            offset = 0
            while True:
                rows = conn.execute(
                    """
                    SELECT * FROM performers
                    WHERE face_count > 0
                    ORDER BY id LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset)
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    yield Performer(**dict(row))
                offset += batch_size

    def search_by_name(self, name: str, limit: int = 20) -> list[Performer]:
        """Search performers by name (case-insensitive partial match)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM performers
                WHERE canonical_name LIKE ?
                ORDER BY face_count DESC
                LIMIT ?
                """,
                (f"%{name}%", limit)
            ).fetchall()
            return [Performer(**dict(row)) for row in rows]


# Convenience function
def open_database(path: str | Path) -> PerformerDatabaseReader:
    """Open a performer database for reading."""
    return PerformerDatabaseReader(path)
