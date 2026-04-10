from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class ChunkMetadata:
    title:      str = ""
    subtitle:   str = ""
    chapter:    str = ""
    subchapter: str = ""
    part:       str = ""
    heading:    str = ""

@dataclass
class Chunk:
    chunk_id:   str = ""
    text:       str = ""
    source:     str = ""      # "federal" | "nevada"
    section:    str = ""      # e.g. "§1381"
    breadcrumb: str = ""      # e.g. "Title 26 > §1381 > (a) > (2) > (B)"
    chunk_type: str = ""      # "flat" | "hierarchy"
    metadata:   ChunkMetadata = field(default_factory=ChunkMetadata)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = asdict(self.metadata)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        meta = ChunkMetadata(**d.pop("metadata", {}))
        return Chunk(**d, metadata=meta)
