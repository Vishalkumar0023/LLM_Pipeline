"""
Dataset Registry Module
=======================
Local-first dataset versioning and registry for LLM fine-tuning pipelines.
Tracks dataset versions, metadata, lineage, and enables rollback.
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class DatasetRegistry:
    """
    Local-first dataset versioning system.

    Stores versioned datasets with metadata, supports diffing between
    versions, listing, loading, and rollback.

    Directory structure:
        registry_dir/
        ├── registry.json       # Version index
        ├── v1.0.0/
        │   ├── data.jsonl      # Dataset file
        │   └── metadata.json   # Version metadata
        ├── v1.1.0/
        │   ├── data.jsonl
        │   └── metadata.json
        └── ...

    Example:
    --------
    >>> registry = DatasetRegistry("./llm_datasets")
    >>> registry.register(training_pairs, version="v1.0.0", description="Initial dataset")
    >>> registry.list_versions()
    >>> old_data = registry.load_version("v1.0.0")
    """

    REGISTRY_FILE = "registry.json"

    def __init__(self, registry_dir: str = "./dataset_registry"):
        """
        Initialize the dataset registry.

        Parameters
        ----------
        registry_dir : str
            Root directory for storing versioned datasets.
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load or create the registry index file."""
        reg_path = self.registry_dir / self.REGISTRY_FILE
        if reg_path.exists():
            with open(reg_path, "r") as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "versions": {},
            "latest": None,
        }

    def _save_registry(self):
        """Persist registry index to disk."""
        reg_path = self.registry_dir / self.REGISTRY_FILE
        with open(reg_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register(
        self,
        data: List[Dict[str, Any]],
        version: str,
        description: str = "",
        source_files: Optional[List[str]] = None,
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new dataset version.

        Parameters
        ----------
        data : list of dict
            Training data samples (JSONL-compatible dicts).
        version : str
            Semantic version string (e.g., "v1.0.0").
        description : str
            Human-readable description of this version.
        source_files : list of str, optional
            Source files used to create this dataset.
        parent_version : str, optional
            Previous version this was derived from.
        tags : list of str, optional
            Tags for this version (e.g., ["production", "medical"]).

        Returns
        -------
        dict
            Version metadata.
        """
        if version in self._registry["versions"]:
            raise ValueError(
                f"Version '{version}' already exists. "
                f"Use a different version string or delete the existing one."
            )

        # Create version directory
        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write data as JSONL
        data_path = version_dir / "data.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for item in data:
                # Strip quality metadata for storage
                clean = {k: v for k, v in item.items() if k != "quality"}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        # Compute data hash
        data_hash = self._compute_hash(data_path)

        # Build metadata
        metadata = {
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "row_count": len(data),
            "data_hash": data_hash,
            "file_size_bytes": data_path.stat().st_size,
            "source_files": source_files or [],
            "parent_version": parent_version,
            "tags": tags or [],
        }

        # Write metadata
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry index
        self._registry["versions"][version] = {
            "created_at": metadata["created_at"],
            "row_count": metadata["row_count"],
            "data_hash": data_hash,
            "description": description,
            "parent_version": parent_version,
            "tags": tags or [],
        }
        self._registry["latest"] = version
        self._save_registry()

        print(f"✅ Registered dataset version: {version}")
        print(f"   • {len(data)} samples, {data_path.stat().st_size:,} bytes")
        print(f"   • Hash: {data_hash[:12]}...")

        return metadata

    def load_version(self, version: str) -> List[Dict[str, Any]]:
        """
        Load a specific dataset version.

        Parameters
        ----------
        version : str
            Version to load.

        Returns
        -------
        list of dict
            Dataset samples.
        """
        if version not in self._registry["versions"]:
            raise ValueError(
                f"Version '{version}' not found. "
                f"Available: {list(self._registry['versions'].keys())}"
            )

        data_path = self.registry_dir / version / "data.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file missing for version: {version}")

        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        return data

    def load_latest(self) -> List[Dict[str, Any]]:
        """Load the latest registered version."""
        if not self._registry["latest"]:
            raise ValueError("No versions registered yet.")
        return self.load_version(self._registry["latest"])

    def get_metadata(self, version: str) -> Dict[str, Any]:
        """Get metadata for a specific version."""
        meta_path = self.registry_dir / version / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for version: {version}")

        with open(meta_path, "r") as f:
            return json.load(f)

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all registered versions with metadata.

        Returns
        -------
        list of dict
            Version info sorted by creation date.
        """
        versions = []
        for ver, info in self._registry["versions"].items():
            versions.append(
                {
                    "version": ver,
                    **info,
                    "is_latest": ver == self._registry.get("latest"),
                }
            )

        # Sort by creation date
        versions.sort(key=lambda v: v.get("created_at", ""))
        return versions

    def diff(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """
        Compute diff between two dataset versions.

        Parameters
        ----------
        version_a : str
            First (older) version.
        version_b : str
            Second (newer) version.

        Returns
        -------
        dict
            Diff report with counts and sample changes.
        """
        data_a = self.load_version(version_a)
        data_b = self.load_version(version_b)

        # Create content hashes for comparison
        hashes_a = {self._hash_item(item): item for item in data_a}
        hashes_b = {self._hash_item(item): item for item in data_b}

        set_a = set(hashes_a.keys())
        set_b = set(hashes_b.keys())

        added_hashes = set_b - set_a
        removed_hashes = set_a - set_b
        unchanged_hashes = set_a & set_b

        diff_report = {
            "version_a": version_a,
            "version_b": version_b,
            "rows_a": len(data_a),
            "rows_b": len(data_b),
            "added": len(added_hashes),
            "removed": len(removed_hashes),
            "unchanged": len(unchanged_hashes),
            "net_change": len(data_b) - len(data_a),
            "sample_added": [hashes_b[h] for h in list(added_hashes)[:3]],
            "sample_removed": [hashes_a[h] for h in list(removed_hashes)[:3]],
        }

        return diff_report

    def delete_version(self, version: str) -> bool:
        """
        Delete a dataset version.

        Parameters
        ----------
        version : str
            Version to delete.

        Returns
        -------
        bool
            True if deleted successfully.
        """
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not found.")

        # Remove directory
        version_dir = self.registry_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Update registry
        del self._registry["versions"][version]
        if self._registry["latest"] == version:
            remaining = list(self._registry["versions"].keys())
            self._registry["latest"] = remaining[-1] if remaining else None
        self._save_registry()

        print(f"🗑️  Deleted version: {version}")
        return True

    def rollback(self, version: str) -> List[Dict[str, Any]]:
        """
        Set a previous version as the latest and return its data.

        Parameters
        ----------
        version : str
            Version to rollback to.

        Returns
        -------
        list of dict
            Data from the rolled-back version.
        """
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not found.")

        self._registry["latest"] = version
        self._save_registry()
        print(f"⏪ Rolled back to version: {version}")
        return self.load_version(version)

    # ─── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _hash_item(item: Dict[str, Any]) -> str:
        """Create a content hash for a single data item."""
        # Hash a deterministic JSON representation
        clean = {k: v for k, v in item.items() if k not in ("quality", "metadata")}
        content = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()

    def print_summary(self) -> None:
        """Print a formatted registry summary."""
        versions = self.list_versions()
        print("=" * 60)
        print("DATASET REGISTRY SUMMARY")
        print("=" * 60)
        print(f"\n📂 Registry: {self.registry_dir}")
        print(f"📊 Total versions: {len(versions)}")
        print(f"🏷️  Latest: {self._registry.get('latest', 'none')}")

        if versions:
            print("\n📋 Versions:")
            for v in versions:
                latest = " ⭐" if v.get("is_latest") else ""
                print(
                    f"   • {v['version']}: {v['row_count']} rows "
                    f"({v.get('description', 'no description')}){latest}"
                )

        print("=" * 60)
