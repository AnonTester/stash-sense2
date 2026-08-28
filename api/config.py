"""Configuration for the face recognition database builder."""
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class StashConfig:
    """Local Stash instance configuration."""
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "StashConfig":
        return cls(
            url=os.environ.get("STASH_URL", "http://localhost:9999"),
            api_key=os.environ.get("STASH_API_KEY", ""),
        )


@dataclass
class StashDBConfig:
    """StashDB API configuration."""
    url: str
    api_key: str
    rate_limit_delay: float = 0.5  # Seconds between requests - EASILY CONFIGURABLE

    @classmethod
    def from_env(cls) -> "StashDBConfig":
        return cls(
            url=os.environ.get("STASHDB_URL", "https://stashdb.org/graphql"),
            api_key=os.environ.get("STASHDB_API_KEY", ""),
            rate_limit_delay=float(os.environ.get("STASHDB_RATE_LIMIT", "0.5")),
        )


@dataclass
class BuilderConfig:
    """Configuration for database building."""
    # Processing limits
    max_images_per_performer: int = 10
    max_performers: int = None  # None = no limit
    batch_size: int = 100
    completeness_threshold: int = 5

    # Quality filters
    min_face_confidence: float = 0.8  # RetinaFace detection confidence threshold
    min_face_size: int = 50  # Minimum face width/height in pixels

    # Output
    version: str = None  # Auto-generated if not specified

    def __post_init__(self):
        if self.version is None:
            self.version = datetime.now().strftime("%Y.%m.%d")


@dataclass
class DatabaseConfig:
    """Configuration for the face recognition database files."""
    data_dir: Path

    # Index file (single usearch index -- buffalo_l produces one embedding
    # per face, replacing the old dual Voyager facenet/arcface pair)
    embedding_index_path: Path = None

    # Local performer index (built from this Stash instance's own performer
    # cover images, kept alongside the main StashDB-derived index -- see
    # local_performer_index.py). Optional -- may not exist yet.
    local_embedding_index_path: Path = None
    local_faces_json_path: Path = None

    # Metadata files (SQLite is primary, JSON kept for compatibility)
    sqlite_db_path: Path = None
    faces_json_path: Path = None
    performers_json_path: Path = None
    manifest_json_path: Path = None

    # embedding_index -> yaw (degrees), for matching.py's steep-angle soft
    # penalty -- optional, may not exist (older dataset published before
    # this feature). See export_db_to_json.py's export_face_yaw_json().
    face_yaw_json_path: Path = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_index_path = self.embedding_index_path or self.data_dir / "face_embeddings.usearch"
        self.local_embedding_index_path = self.local_embedding_index_path or self.data_dir / "local_embeddings.usearch"
        self.local_faces_json_path = self.local_faces_json_path or self.data_dir / "local_faces.json"
        self.sqlite_db_path = self.sqlite_db_path or self.data_dir / "performers.db"
        self.faces_json_path = self.faces_json_path or self.data_dir / "faces.json"
        self.performers_json_path = self.performers_json_path or self.data_dir / "performers.json"
        self.manifest_json_path = self.manifest_json_path or self.data_dir / "manifest.json"
        self.face_yaw_json_path = self.face_yaw_json_path or self.data_dir / "face_yaw.json"


# Embedding dimensions
EMBEDDING_DIM = 512

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Stash-box endpoints (for universal ID generation)
STASHBOX_ENDPOINTS = {
    "https://stashdb.org/graphql": "stashdb.org",
    "https://pmvstash.org/graphql": "pmvstash.org",
    "https://fansdb.cc/graphql": "fansdb.cc",
    "https://javstash.org/graphql": "javstash.org",
    "https://theporndb.net/graphql": "theporndb.net",  # Uses REST API, not GraphQL
}


def get_stashbox_shortname(endpoint_url: str) -> str:
    """Convert a stash-box GraphQL URL to a short name for universal IDs."""
    return STASHBOX_ENDPOINTS.get(endpoint_url, endpoint_url.replace("https://", "").replace("/graphql", ""))
