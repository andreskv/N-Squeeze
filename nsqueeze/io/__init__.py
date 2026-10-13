"""Fetchers and storage. Every fetcher writes to the point-in-time store."""

from nsqueeze.io.pit import PITStore

__all__ = ["PITStore"]
