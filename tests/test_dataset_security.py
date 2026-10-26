from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_cybench_subset as prep


class DatasetArchiveSecurityTests(unittest.TestCase):
    def test_zip_extraction_skips_links_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            archive = out_dir / "release" / "sample.zip"
            archive.parent.mkdir()
            link = zipfile.ZipInfo("link.txt")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("safe/file.txt", "safe payload")
                zf.writestr("../escape.txt", "escape payload")
                zf.writestr(link, "../../outside.txt")

            extracted = prep.extract_archive(archive, out_dir)
            self.assertEqual(
                [path.relative_to(out_dir).as_posix() for path in extracted],
                ["_extracted/release__sample.zip/safe/file.txt"],
            )
            self.assertFalse((out_dir.parent / "escape.txt").exists())

    def test_zip_extraction_enforces_member_limit_while_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            archive = out_dir / "release" / "large.zip"
            archive.parent.mkdir()
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("large.bin", b"0123456789")

            with patch.object(prep, "MAX_ARCHIVE_MEMBER_BYTES", 8):
                with self.assertRaisesRegex(RuntimeError, "size limit"):
                    prep.extract_archive(archive, out_dir)

            extracted_path = prep.archive_extract_dir(archive, out_dir) / "large.bin"
            self.assertFalse(extracted_path.exists())


if __name__ == "__main__":
    unittest.main()
