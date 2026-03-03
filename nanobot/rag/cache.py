"""Search cache management with LRU + TTL."""

import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar('T')


class SearchCache(Generic[T]):
    """
    Search result cache with LRU eviction and TTL expiration.

    Uses OrderedDict for LRU ordering (oldest items at front).
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries before LRU eviction
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def _cleanup(self) -> None:
        """Remove expired and excess entries from cache."""
        now = time.time()

        # Remove expired entries
        expired_keys = [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl_seconds]
        for k in expired_keys:
            del self._cache[k]

        # Remove excess entries (LRU - oldest first)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def get(self, key: str) -> T | None:
        """
        Get value from cache with TTL check and LRU touch.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key in self._cache:
            ts, value = self._cache[key]
            if time.time() - ts > self._ttl_seconds:
                del self._cache[key]
                return None
            # Move to end (LRU - most recently used)
            self._cache.move_to_end(key)
            return value
        return None

    def set(self, key: str, value: T) -> None:
        """
        Set value in cache with current timestamp and LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Cleanup before adding (evict if necessary)
        self._cleanup()
        self._cache[key] = (time.time(), value)
        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)


class SearchCacheManager:
    """Manages multiple search caches."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of entries per cache
            ttl_seconds: TTL for cache entries in seconds
        """
        self._advanced_cache: SearchCache = SearchCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self._basic_cache: SearchCache = SearchCache(max_size=max_size, ttl_seconds=ttl_seconds)

    @property
    def advanced(self) -> SearchCache:
        """Get advanced search cache."""
        return self._advanced_cache

    @property
    def basic(self) -> SearchCache:
        """Get basic search cache."""
        return self._basic_cache

    def clear_all(self) -> None:
        """Clear all caches."""
        self._advanced_cache.clear()
        self._basic_cache.clear()
