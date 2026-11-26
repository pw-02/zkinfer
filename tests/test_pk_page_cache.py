import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zkinfer.profiling.pk_page_cache import prepare_pk_page_cache


class PKPageCacheTests(unittest.TestCase):
    def test_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pk"
            path.write_bytes(b"abc")
            with patch("zkinfer.profiling.pk_page_cache.cached_bytes", return_value=3), \
                 patch("os.fsync") as sync:
                result = prepare_pk_page_cache(path)
                self.assertEqual(result["pk_page_cache_eviction_status"], "disabled")
                sync.assert_not_called()
                self.assertEqual(path.read_bytes(), b"abc")

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "requires posix_fadvise")
    def test_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pk"
            path.write_bytes(b"abc")
            with patch("zkinfer.profiling.pk_page_cache.sys.platform", "linux"), \
                 patch("zkinfer.profiling.pk_page_cache.cached_bytes", side_effect=[3, 0]), \
                 patch("os.fsync") as sync, patch("os.posix_fadvise") as advise:
                result = prepare_pk_page_cache(path, evict=True)
                sync.assert_called_once()
                advise.assert_called_once()
                self.assertEqual(advise.call_args.args[1:], (0, 0, os.POSIX_FADV_DONTNEED))
                self.assertEqual(result["pk_page_cache_cached_bytes_before"], 3)
                self.assertEqual(result["pk_page_cache_cached_bytes_after"], 0)
                self.assertEqual(result["pk_page_cache_eviction_status"], "advised")
                self.assertEqual(path.read_bytes(), b"abc")

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "requires posix_fadvise")
    def test_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pk"
            path.write_bytes(b"abc")
            with patch("zkinfer.profiling.pk_page_cache.sys.platform", "linux"), \
                 patch("zkinfer.profiling.pk_page_cache.cached_bytes", return_value=None), \
                 patch("os.fsync", side_effect=OSError("flush failed")):
                with self.assertRaisesRegex(RuntimeError, "Failed to flush/advise"):
                    prepare_pk_page_cache(path, evict=True)
