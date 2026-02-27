#!/usr/bin/env python3
"""Test the sqlite-vec compatibility fixes in store.py."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_sqlite_vec_compatibility():
    """Test that sqlite-vec API compatibility works correctly."""
    print("1. Testing sqlite-vec API compatibility...")

    # Test that we can handle both loadable_path() and extension_path()
    try:
        import sqlite_vec

        # Check which methods are available
        available_methods = []
        if hasattr(sqlite_vec, 'loadable_path'):
            available_methods.append('loadable_path')
            try:
                path = sqlite_vec.loadable_path()
                print(f"   ✓ loadable_path() available, returns: {path}")
            except Exception as e:
                print(f"   ✗ loadable_path() exists but failed: {e}")

        if hasattr(sqlite_vec, 'extension_path'):
            available_methods.append('extension_path')
            try:
                path = sqlite_vec.extension_path()
                print(f"   ✓ extension_path() available, returns: {path}")
            except Exception as e:
                print(f"   ✗ extension_path() exists but failed: {e}")

        if hasattr(sqlite_vec, 'path'):
            available_methods.append('path')
            print(f"   ✓ path attribute available: {sqlite_vec.path}")

        if not available_methods:
            print(f"   ! No known sqlite-vec path methods found")
        else:
            print(f"   ✓ Found methods: {', '.join(available_methods)}")

    except ImportError:
        print("   ! sqlite-vec not installed, skipping API tests")

    print()

async def test_document_store_init():
    """Test that DocumentStore initializes correctly even without sqlite-vec."""
    print("2. Testing DocumentStore initialization...")

    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"

        # Create a mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.dimensions = 384

        # Import here to test with potential mocks
        from nanobot.rag.store import DocumentStore

        # Test 1: Normal initialization (with whatever sqlite-vec support we have)
        print("   Testing normal initialization...")
        try:
            store = DocumentStore(db_path, mock_embedding_provider)
            # Force initialization
            db = store._get_db()
            print(f"   ✓ DocumentStore initialized successfully")
            print(f"   ✓ Vector enabled: {store._vector_enabled}")
        except Exception as e:
            print(f"   ✗ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return

        print()

async def test_document_store_functionality():
    """Test that DocumentStore works (at least with FTS)."""
    print("3. Testing DocumentStore functionality...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create a simple test document
        test_file = docs_dir / "test.txt"
        test_file.write_text("""
        This is a test document about artificial intelligence.
        Machine learning is a subset of AI.
        Deep learning is a subset of machine learning.
        Neural networks are used in deep learning.
        """)

        # Use real embedding provider if possible
        try:
            from nanobot.rag import DocumentStore, SentenceTransformerEmbeddingProvider
            embedding_provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"   ! Could not create real embedding provider: {e}")
            print("   ! Using mock embedding provider")
            # Create mock
            from unittest.mock import MagicMock
            embedding_provider = MagicMock()
            embedding_provider.dimensions = 384
            embedding_provider.embed.return_value = [0.1] * 384
            embedding_provider.embed_batch.return_value = [[0.1] * 384]

        store = DocumentStore(db_path, embedding_provider)

        # Test scan and index
        print("   Testing scan_and_index...")
        try:
            stats = await store.scan_and_index(docs_dir, chunk_size=100, chunk_overlap=0)
            print(f"   ✓ scan_and_index complete: {stats}")
        except Exception as e:
            print(f"   ✗ scan_and_index failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with tests even if indexing failed

        # Test get_stats
        print("   Testing get_stats...")
        try:
            stats = store.get_stats()
            print(f"   ✓ get_stats: {stats}")
        except Exception as e:
            print(f"   ✗ get_stats failed: {e}")

        # Test search
        print("   Testing search...")
        try:
            results = await store.search("machine learning", top_k=3)
            print(f"   ✓ search returned {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"     [{i}] {result.filename} (score: {result.score:.2f}, {result.source})")
        except Exception as e:
            print(f"   ✗ search failed: {e}")
            import traceback
            traceback.print_exc()

    print()

def test_enable_load_extension_check():
    """Test that we correctly check for enable_load_extension attribute."""
    print("4. Testing enable_load_extension check...")

    import sqlite3

    # Create a simple in-memory DB
    conn = sqlite3.connect(":memory:")

    # Check if enable_load_extension exists
    has_enable = hasattr(conn, 'enable_load_extension')
    print(f"   ✓ sqlite3.Connection has enable_load_extension: {has_enable}")

    conn.close()

    print()

async def main():
    print("Testing sqlite-vec compatibility fixes\n")
    print("=" * 50)

    test_sqlite_vec_compatibility()
    await test_document_store_init()
    await test_document_store_functionality()
    test_enable_load_extension_check()

    print("=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
