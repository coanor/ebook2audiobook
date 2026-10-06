import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib import core


class OutputNameTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.book = self.root / '置身事内.epub'
        self.book.touch()
        self.addCleanup(patch.stopall)
        for name in ('tmp_dir', 'voices_dir', 'models_dir'):
            patch.object(core, name, str(self.root / name)).start()
        patch.object(core.tempfile, 'gettempdir', return_value=str(self.root)).start()
        patch.object(core, 'cleanup_models_cache').start()
        patch.object(core, 'delete_unused_tmp_dirs').start()
        # Exercise real input/session/name preparation, stopping before Calibre
        # or model loading. The export tests cover the resulting audio and VTT.
        patch.object(core, 'prepare_dirs', return_value=False).start()

    def prepare(self, engine, text=None, **options):
        session = dict(id='test-session', cancellation_requested=False)
        args = dict(id=session['id'], ebook_mode=core.ebook_modes['SINGLE'],
                    ebook_src=str(self.book), language='zho', device='cpu',
                    is_gui_process=False, tts_engine=engine, fine_tuned='internal',
                    custom_model=None, output_format='m4b', output_channel='mono',
                    output_split=True, output_split_hours='chapters',
                    output_dir=str(self.root), bark_text_temp=.7, bark_waveform_temp=.7)
        for name in ('temperature', 'length_penalty', 'num_beams', 'repetition_penalty',
                     'top_k', 'top_p', 'speed', 'enable_text_splitting'):
            args['xtts_' + name] = core.default_engine_settings['xtts'][name]
        if text is not None:
            args.update(ebook_mode=core.ebook_modes['TEXT'], ebook_textarea=text)
        args.update(options)
        with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)):
            error, passed = core.convert_ebook(args)
        self.assertFalse(passed)
        self.assertIn('not removed due to failure', error)
        return session

    def test_book_outputs_include_engine_and_keep_existing_process_directory(self):
        expected_process = self.root / 'tmp_dir' / 'proc-test-session' / hashlib.md5('置身事内'.encode()).hexdigest()
        old = self.root / '置身事内.m4b'
        old.write_bytes(b'previous output')
        for engine in ('xtts', 'cosyvoice', 'qwen3', 'indextts'):
            with self.subTest(engine=engine):
                session = self.prepare(engine)
                self.assertEqual(Path(session['final_name']).name, f'置身事内_{engine}.m4b')
                self.assertEqual(Path(session['process_dir']), expected_process)
                self.assertEqual(old.read_bytes(), b'previous output')

    def test_raw_text_outputs_include_engine_and_preserve_text_session_prefix(self):
        for engine in ('cosyvoice', 'qwen3'):
            with self.subTest(engine=engine):
                session = self.prepare(engine, text='五千年的文明。')
                self.assertEqual(Path(session['final_name']).name,
                                 f'五千年的文明。_test-session_{engine}.m4b')

    def test_sample_translation_and_format_suffixes_are_preserved(self):
        session = self.prepare('qwen3', output_format='wav', translate_enabled=True,
                               translate='eng', chapter_selection={'number': 4, 'title': '第四章'})
        self.assertEqual(Path(session['final_name']).name, '置身事内_eng_sample_qwen3.wav')
        expected = hashlib.md5('置身事内_eng_sample'.encode()).hexdigest()
        self.assertEqual(Path(session['process_dir']).name, expected)

    def test_gui_uses_same_engine_name_without_absolute_output_directory(self):
        session = self.prepare('cosyvoice', is_gui_process=True)
        self.assertEqual(session['final_name'], '置身事内_cosyvoice.m4b')


if __name__ == '__main__':
    unittest.main()
