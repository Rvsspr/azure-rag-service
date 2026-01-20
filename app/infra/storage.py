from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

"""
app/infra/storage.py

Storage abstraction used by the application. Provides a simple interface
for uploading/downloading/deleting/listing files and two implementations:

- LocalStorage: filesystem-backed storage for local development/tests.
- AzureBlobStorage: Azure Blob Storage-backed implementation.

This file is intentionally conservative and dependency-safe: Azure SDK is
optional and only required when using AzureBlobStorage.

Adjust to match app design.md and other modules as needed.
"""


# Optional import for Azure Blob Storage
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
except Exception:
    BlobServiceClient = None  # type: ignore
    ContainerClient = None  # type: ignore


class StorageError(Exception):
    """Generic storage error."""


class NotFoundError(StorageError):
    """Raised when an object is not found in storage."""


class Storage(ABC):
    """Abstract storage interface used by the application."""

    @abstractmethod
    def upload(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        """
        Upload bytes to storage.

        Returns:
            str: A URI or path that can be used to access the uploaded object.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download bytes for a given key. Raises NotFoundError if missing."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete object identified by key."""
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> Iterable[str]:
        """Yield keys under the optional prefix."""
        raise NotImplementedError

    @abstractmethod
    def url(self, key: str) -> str:
        """Return a public or signed URL for the given key if supported, otherwise a path."""
        raise NotImplementedError


@dataclass
class LocalStorage(Storage):
    """
    Local filesystem storage implementation. Useful for development and tests.

    root: directory where objects are stored. Keys are treated as relative paths.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Prevent escaping the storage root
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise StorageError("Invalid key: attempted path traversal")
        return candidate

    def upload(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            fh.write(data)
        return str(p)

    def download(self, key: str) -> bytes:
        p = self._path_for(key)
        if not p.exists():
            raise NotFoundError(f"Key not found: {key}")
        return p.read_bytes()

    def delete(self, key: str) -> None:
        p = self._path_for(key)
        try:
            p.unlink()
        except FileNotFoundError:
            raise NotFoundError(f"Key not found: {key}")

    def list_keys(self, prefix: Optional[str] = None) -> Iterable[str]:
        root = self.root
        for path in root.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(root)).replace("\\", "/")
                if prefix is None or rel.startswith(prefix):
                    yield rel

    def url(self, key: str) -> str:
        # For local storage, return file path. Higher-level code may serve files.
        return self._path_for(key).as_uri()


@dataclass
class AzureBlobStorage(Storage):
    """
    Azure Blob Storage implementation.

    Required environment variables/config:
    - AZURE_BLOB_CONNECTION_STRING or pass connection_string
    - container: container name to operate in (will be created if missing)
    """

    container: str
    connection_string: Optional[str] = None
    client: Optional[BlobServiceClient] = None
    _container_client: Optional[ContainerClient] = None

    def __post_init__(self) -> None:
        if BlobServiceClient is None:
            raise ImportError(
                "azure-storage-blob is required for AzureBlobStorage. "
                "Install with: pip install azure-storage-blob"
            )
        conn = self.connection_string or os.getenv("AZURE_BLOB_CONNECTION_STRING")
        if not conn:
            raise ValueError("Azure connection string not provided (connection_string or AZURE_BLOB_CONNECTION_STRING)")
        self.client = BlobServiceClient.from_connection_string(conn)
        self._container_client = self.client.get_container_client(self.container)
        try:
            self._container_client.create_container()
        except Exception:
            # container may already exist; ignore error
            pass

    def _ensure_container(self) -> ContainerClient:
        if self._container_client is None:
            assert self.client is not None
            self._container_client = self.client.get_container_client(self.container)
        return self._container_client

    def upload(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        container = self._ensure_container()
        blob = container.get_blob_client(key)
        kwargs = {}
        if content_type:
            kwargs["content_settings"] = {"content_type": content_type}
        try:
            # azure SDK expects ContentSettings object; using bytes upload and headers minimally
            blob.upload_blob(data, overwrite=True)
        except Exception as exc:
            raise StorageError(f"Azure upload failed: {exc}") from exc
        return self.url(key)

    def download(self, key: str) -> bytes:
        container = self._ensure_container()
        blob = container.get_blob_client(key)
        try:
            stream = blob.download_blob()
            return stream.readall()
        except Exception as exc:
            # Normalize not found vs other errors
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                raise NotFoundError(f"Key not found: {key}") from exc
            raise StorageError(f"Azure download failed: {exc}") from exc

    def delete(self, key: str) -> None:
        container = self._ensure_container()
        blob = container.get_blob_client(key)
        try:
            blob.delete_blob()
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                raise NotFoundError(f"Key not found: {key}") from exc
            raise StorageError(f"Azure delete failed: {exc}") from exc

    def list_keys(self, prefix: Optional[str] = None) -> Iterable[str]:
        container = self._ensure_container()
        try:
            for blob in container.list_blobs(name_starts_with=prefix):
                yield blob.name
        except Exception as exc:
            raise StorageError(f"Azure list failed: {exc}") from exc

    def url(self, key: str) -> str:
        # Construct blob url. If using SAS or custom domain, higher-level code should override.
        if not self.client:
            raise StorageError("Azure client not initialized")
        account_url = self.client.url.rstrip("/")
        return f"{account_url}/{self.container}/{key}"


# Factory helper
def get_storage_from_env() -> Storage:
    """
    Create a Storage implementation based on environment variables.

    Use AZURE_BLOB_CONTAINER to enable Azure; otherwise use local filesystem storage
    at STORAGE_ROOT (defaults to ./data).
    """
    container = os.getenv("AZURE_BLOB_CONTAINER")
    if container:
        conn = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        return AzureBlobStorage(container=container, connection_string=conn)
    root = os.getenv("STORAGE_ROOT", "./data")
    return LocalStorage(Path(root))