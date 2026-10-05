import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from lib.classes.cli_session import select_book_session


class RememberedBookSessionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.cache = self.root / 'tmp'
        self.book = self.root / 'book.epub'
        self.book.write_bytes(b'original book')

    def select(self, book=None, **options):
        return select_book_session(book or self.book, self.cache, 'book', 'zho', **options)

    def legacy_session(self, content=b'original book', modified=1, process_name='book'):
        session_id = str(uuid.uuid4())
        process_key = hashlib.md5(process_name.encode()).hexdigest()
        process = self.cache / f'proc-{session_id}' / process_key
        process.mkdir(parents=True)
        (process / 'checksum').write_text(hashlib.sha256(content).hexdigest())
        (process / 'sentences.flac').write_bytes(b'saved progress')
        os.utime(process, (modified, modified))
        return session_id, process

    def test_rerun_reuses_session_and_keeps_saved_progress(self):
        first, reused = self.select()
        self.assertFalse(reused)
        checkpoint = self.cache / f'proc-{first}' / 'checkpoint'
        checkpoint.write_text('sentence 42')
        second, reused = self.select()
        self.assertTrue(reused)
        self.assertEqual(first, second)
        self.assertEqual(checkpoint.read_text(), 'sentence 42')

    def test_adopts_latest_matching_legacy_session_and_remembers_it(self):
        self.legacy_session(modified=1)
        latest, process = self.legacy_session(modified=2)
        self.legacy_session(content=b'different book', modified=3)
        selected, reused = self.select()
        self.assertEqual(selected, latest)
        self.assertTrue(reused)
        self.assertEqual((process / 'sentences.flac').read_bytes(), b'saved progress')
        self.assertEqual(self.select(), (latest, True))

    def test_books_with_same_basename_get_separate_sessions(self):
        first, _ = self.select()
        other = self.root / 'other' / 'book.epub'
        other.parent.mkdir()
        other.write_bytes(b'another book')
        second, _ = self.select(book=other)
        self.assertNotEqual(first, second)
        self.assertEqual(self.select(), (first, True))

    def test_book_edits_keep_session_for_existing_core_realignment(self):
        first, _ = self.select()
        self.book.write_bytes(b'updated book')
        self.assertEqual(self.select(), (first, True))

    def test_translated_book_does_not_adopt_untranslated_cache(self):
        untranslated, _ = self.legacy_session()
        translated, _ = select_book_session(self.book, self.cache, 'book_eng', 'zho', 'eng')
        self.assertNotEqual(untranslated, translated)
        self.assertEqual(self.select(), (untranslated, True))
        self.assertEqual(select_book_session(self.book, self.cache, 'book_eng', 'zho', 'eng'),
                         (translated, True))

    def test_new_session_preserves_old_progress_and_becomes_default(self):
        old, _ = self.select()
        checkpoint = self.cache / f'proc-{old}' / 'checkpoint'
        checkpoint.write_text('old progress')
        fresh, reused = self.select(new_session=True)
        self.assertFalse(reused)
        self.assertNotEqual(old, fresh)
        self.assertEqual(checkpoint.read_text(), 'old progress')
        self.assertEqual(self.select(), (fresh, True))

    def test_deleted_session_is_replaced(self):
        old, _ = self.select()
        (self.cache / f'proc-{old}').rmdir()
        fresh, reused = self.select()
        self.assertFalse(reused)
        self.assertNotEqual(old, fresh)

    def test_corrupt_pointer_recovers_from_matching_legacy_progress(self):
        selected, _ = self.select()
        index = next((self.cache / 'cli_sessions').glob('*.json'))
        index.write_text('{interrupted write')
        legacy, _ = self.legacy_session()
        self.assertEqual(self.select(), (legacy, True))
        self.assertEqual(json.loads(index.read_text())['session_id'], legacy)
        self.assertTrue((self.cache / f'proc-{selected}').exists())

    def test_explicit_session_is_remembered_and_invalid_id_fails(self):
        selected, _ = self.select()
        chosen = str(uuid.uuid4())
        (self.cache / f'proc-{chosen}').mkdir()
        self.assertEqual(self.select(explicit_session=chosen), (chosen, True))
        self.assertEqual(self.select(), (chosen, True))
        with self.assertRaisesRegex(ValueError, 'Session expired'):
            self.select(explicit_session=str(uuid.uuid4()))
        self.assertTrue((self.cache / f'proc-{selected}').exists())


if __name__ == '__main__':
    unittest.main()
