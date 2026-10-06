import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from lib import core


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required')
class IncrementalChapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chapters = self.root / 'chapters'
        self.sentences = self.chapters / 'sentences'
        self.output = self.root / 'output'
        self.chapters.mkdir()
        self.output.mkdir()
        self.blocks = [self.block('a', 'First heading', 'First'),
                       self.block('skip', 'Skipped text', 'First', keep=False),
                       self.block('b', 'First body', 'First'),
                       self.block('c', 'Second heading', 'Second'),
                       self.block('d', 'Second body', 'Second')]
        self.session = dict(id='incremental', is_gui_process=False, cancellation_requested=False,
                            ebook='book.epub', filename_noext='book', language='eng', voice=None,
                            blocks_current={'blocks': self.blocks, 'block_resume': 0, 'sentence_resume': 0},
                            blocks_orig={'blocks': copy.deepcopy(self.blocks)}, blocks_saved={},
                            process_dir=str(self.root), chapters_dir=str(self.chapters),
                            sentences_dir=str(self.sentences), audiobooks_dir=str(self.output),
                            output_split=True, output_split_hours='chapters',
                            output_format='m4b', output_channel='mono', cover=None,
                            metadata={'title': 'Book', 'creator': 'Author'}, final_name='book_xtts.m4b', tts_engine='xtts')
        self.first = self.output / 'book_xtts_chapter1_First.m4b'
        self.second = self.output / 'book_xtts_chapter2_Second.m4b'
        self.addCleanup(patch.stopall)
        patch.object(core, 'context', SimpleNamespace(get_session=lambda _: self.session)).start()
        for method in ('show_alert', 'save_db_stamp', 'save_json_blocks', 'unload_tts_manager'):
            patch.object(core, method).start()

    @staticmethod
    def block(identifier, text, chapter, keep=True):
        return dict(id=identifier, text=text, sentences=[text], keep=keep,
                    book_chapter={'key': chapter, 'title': chapter})

    @staticmethod
    def write_audio(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wave = np.sin(np.arange(12000) * 2 * np.pi * 220 / 24000).astype('float32') * 0.1
        sf.write(str(path), wave, 24000)

    def complete_block(self, block):
        self.write_audio(self.chapters / f"{block['id']}.flac")
        self.write_audio(self.sentences / block['id'] / '0.flac')

    def test_chapter_available_before_next_chapter_and_reused_after_failed_conversion(self):
        generated = []

        def synthesize(path, sentence, **kwargs):
            generated.append(sentence)
            if sentence == 'First body':
                self.assertFalse(self.first.exists(), 'Publish only after the whole book chapter')
            if sentence == 'Second heading':
                self.assertTrue(self.first.is_file(), 'First output must be ready before continuing')
                self.assertAlmostEqual(core.get_audio_duration(str(self.first)), 1.0, delta=0.06)
                subtitles = self.first.with_suffix('.vtt').read_text()
                self.assertIn('First body', subtitles)
                self.assertNotIn('Second heading', subtitles)
                self.assertNotIn('Skipped text', subtitles)
            if sentence == 'Second body':
                return False, 'Simulated interruption'
            self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentence2audio=synthesize)):
            self.assertFalse(core.convert_chapters2audio('incremental'))
        self.assertTrue(self.first.exists())
        self.assertFalse(self.second.exists())
        first_stamp = self.first.stat().st_mtime_ns
        first_data = self.first.read_bytes()
        generated.clear()

        def resume(path, sentence, **kwargs):
            self.assertTrue(self.first.exists())
            generated.append(sentence)
            self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentence2audio=resume)):
            self.assertTrue(core.convert_chapters2audio('incremental'))
        self.assertEqual(generated, ['Second body'])
        self.assertEqual(self.first.stat().st_mtime_ns, first_stamp)
        self.assertEqual(self.first.read_bytes(), first_data)
        self.assertTrue(self.second.exists())
        second_vtt = self.second.with_suffix('.vtt').read_text()
        self.assertIn('Second body', second_vtt)
        self.assertNotIn('First body', second_vtt)
        self.assertIn('00:00:00.000', second_vtt)
        with patch.object(core, 'assemble_audio_chunks') as assemble:
            self.assertEqual(core.combine_audio_chapters('incremental'), [str(self.first), str(self.second)])
            assemble.assert_not_called()

    def test_incremental_number_padding_uses_full_book_and_ignores_unfinished_chapters(self):
        blocks = [self.block(str(i), f'Text {i}', f'Chapter {i}') for i in range(1, 11)]
        self.session['blocks_current']['blocks'] = blocks
        self.session['blocks_orig']['blocks'] = copy.deepcopy(blocks)
        self.complete_block(blocks[0])
        first = core.combine_audio_chapters('incremental', chapter_index=0)
        self.assertEqual([Path(path).name for path in first], ['book_xtts_chapter01_Chapter_1.m4b'])
        self.complete_block(blocks[8])
        ninth = core.combine_audio_chapters('incremental', chapter_index=8)
        self.assertEqual([Path(path).name for path in ninth], ['book_xtts_chapter09_Chapter_9.m4b'])
        subtitles = Path(ninth[0]).with_suffix('.vtt').read_text()
        self.assertIn('Text 9', subtitles)
        self.assertNotIn('Text 1', subtitles)

    def test_qwen_batches_keep_sentence_files_and_export_before_next_chapter(self):
        self.blocks[0]['sentences'] = [f'First sentence {i}' for i in range(5)]
        self.session.update(tts_engine='qwen3', device='cuda', tts_batch_size=4)
        batches = []

        def batch(items, **kwargs):
            batches.append([sentence for _, sentence in items])
            if items[0][1] == 'Second heading':
                self.assertTrue(self.first.exists())
            for path, _ in items:
                self.write_audio(path)
            return True, None

        manager = SimpleNamespace(convert_sentences2audio=batch)
        with patch.object(core, 'TTSManager', return_value=manager):
            self.assertTrue(core.convert_chapters2audio('incremental'))
        self.assertEqual(batches[:2], [[f'First sentence {i}' for i in range(4)], ['First sentence 4']])
        self.assertEqual(len(list((self.sentences / 'a').glob('*.flac'))), 5)
        subtitles = self.first.with_suffix('.vtt').read_text()
        for i in range(5):
            self.assertIn(f'First sentence {i}', subtitles)

    def test_qwen_resume_reuses_batch_files_ahead_of_checkpoint(self):
        self.blocks[0]['sentences'] = [f'Sentence {i}' for i in range(5)]
        self.session.update(tts_engine='qwen3', device='cuda', tts_batch_size=4)
        self.session['blocks_saved'] = copy.deepcopy(self.session['blocks_current'])
        for i in range(4):
            self.write_audio(self.sentences / 'a' / f'{i}.flac')
        cached_stamp = (self.sentences / 'a' / '0.flac').stat().st_mtime_ns
        batches = []

        def batch(items, **kwargs):
            batches.extend(sentence for _, sentence in items)
            for path, _ in items:
                self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentences2audio=batch)):
            self.assertTrue(core.convert_chapters2audio('incremental'))
        self.assertEqual(batches[0], 'Sentence 4')
        self.assertNotIn('Sentence 0', batches)
        self.assertEqual((self.sentences / 'a' / '0.flac').stat().st_mtime_ns, cached_stamp)

    def test_qwen_interrupted_first_batch_retains_finished_files_on_resume(self):
        self.blocks[0]['sentences'] = [f'Sentence {i}' for i in range(5)]
        self.session.update(tts_engine='qwen3', device='cuda', tts_batch_size=4)

        def interrupted(items, **kwargs):
            for path, _ in items[:2]:
                self.write_audio(path)
            return False, 'Interrupted while publishing the batch'

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentences2audio=interrupted)):
            self.assertFalse(core.convert_chapters2audio('incremental'))
        self.assertEqual(self.session['blocks_current']['sentence_resume'], 0)
        generated = []

        def resume(items, **kwargs):
            generated.extend(sentence for _, sentence in items)
            for path, _ in items:
                self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentences2audio=resume)):
            self.assertTrue(core.convert_chapters2audio('incremental'))
        self.assertEqual(generated[:3], ['Sentence 2', 'Sentence 3', 'Sentence 4'])
        self.assertNotIn('Sentence 0', generated)
        self.assertNotIn('Sentence 1', generated)

    def test_resume_exports_previously_cached_chapter_before_synthesizing_more(self):
        for block in self.blocks[:3]:
            if block['keep']:
                self.complete_block(block)
        self.session['blocks_current']['block_resume'] = 3
        self.session['blocks_saved'] = copy.deepcopy(self.session['blocks_current'])

        def interrupted(path, sentence, **kwargs):
            self.assertEqual(sentence, 'Second heading')
            self.assertTrue(self.first.exists())
            return False, 'Simulated interruption'

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentence2audio=interrupted)):
            self.assertFalse(core.convert_chapters2audio('incremental'))
        self.assertTrue(self.first.exists())
        self.assertFalse(self.second.exists())

    def test_resume_reuses_export_after_database_reload_and_identical_cover_rewrite(self):
        from PIL import Image

        cover = self.root / 'cover.jpg'
        Image.new('RGB', (16, 16), 'blue').save(cover)
        self.session['cover'] = str(cover)
        for block in self.blocks[:3]:
            if block['keep']:
                self.complete_block(block)
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
        cover.write_bytes(cover.read_bytes())
        # SQLite reloads omit chapter labels; editor expansion can also change.
        for block in self.blocks:
            block.pop('book_chapter')
            block['expand'] = True
        with patch.object(core, 'assemble_audio_chunks') as assemble:
            self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
            assemble.assert_not_called()

    def test_adding_engine_name_reexports_cached_audio_and_keeps_old_output(self):
        for block in self.blocks[:3]:
            if block['keep']:
                self.complete_block(block)
        self.session['final_name'] = 'book.m4b'
        old = self.output / 'book_chapter1_First.m4b'
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(old)])
        old_data = old.read_bytes()
        source = self.chapters / 'a.flac'
        source_stamp = source.stat().st_mtime_ns
        self.session['final_name'] = 'book_xtts.m4b'
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
        self.assertEqual(old.read_bytes(), old_data)
        self.assertEqual(source.stat().st_mtime_ns, source_stamp)
        self.assertTrue(self.first.with_suffix('.vtt').is_file())
        with patch.object(core, 'assemble_audio_chunks') as assemble:
            self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
            assemble.assert_not_called()

    def test_resplit_resume_regenerates_changed_audio_and_reuses_unchanged_blocks(self):
        first_text = '从 1940 年开始，这本书继续广泛的印刷发行。'
        unchanged_text = '这个文本块的断句已经正确。'
        last_text = '最后一个文本块尚未完成。'
        blocks = [self.block('a', first_text, 'First'),
                  self.block('b', unchanged_text, 'Second'),
                  self.block('c', last_text, 'Third')]
        blocks[0]['sentences'] = ['从 1940 年', '开始，这本书继续广泛的印刷发行。']
        self.session.update(language='zho', tts_engine='xtts', status=core.status_tags['CONVERTING'],
                            blocks_current=dict(blocks=blocks, block_resume=2, sentence_resume=0),
                            blocks_saved=dict(blocks=copy.deepcopy(blocks)),
                            blocks_orig=dict(blocks=copy.deepcopy(blocks)),
                            blocks_current_db=str(self.root / 'blocks.db'))
        for block in blocks[:2]:
            self.complete_block(block)
        self.write_audio(self.sentences / 'a' / '1.flac')
        unchanged_stamp = (self.chapters / 'b.flac').stat().st_mtime_ns
        generated = []

        def synthesize(path, sentence, **kwargs):
            generated.append(sentence)
            if sentence == last_text:
                return False, 'Simulated interruption after migration'
            self.assertEqual(sentence, first_text)
            self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentence2audio=synthesize)):
            _, success = core.finalize_audiobook('incremental')
        self.assertFalse(success)
        self.assertEqual(generated, [first_text, last_text])
        self.assertEqual((self.chapters / 'b.flac').stat().st_mtime_ns, unchanged_stamp)
        self.assertFalse((self.sentences / 'a' / '1.flac').exists())
        self.assertIn(first_text, self.first.with_suffix('.vtt').read_text())
        first_stamp = self.first.stat().st_mtime_ns
        generated.clear()
        self.session['blocks_current'] = core.load_db_blocks(self.session['blocks_current_db'])

        def resume(path, sentence, **kwargs):
            generated.append(sentence)
            self.write_audio(path)
            return True, None

        with patch.object(core, 'TTSManager', return_value=SimpleNamespace(convert_sentence2audio=resume)):
            self.assertTrue(core.convert_chapters2audio('incremental'))
        self.assertEqual(generated, [last_text])
        self.assertEqual(self.first.stat().st_mtime_ns, first_stamp)

    def test_failed_reexport_keeps_previous_output_and_missing_subtitle_is_rebuilt(self):
        for block in self.blocks[:3]:
            if block['keep']:
                self.complete_block(block)
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
        first_data = self.first.read_bytes()
        subtitle_data = self.first.with_suffix('.vtt').read_bytes()
        manifest = (self.root / 'chapter_export_1.json').read_bytes()
        self.write_audio(self.chapters / 'a.flac')
        with patch.object(core, 'build_vtt_file', return_value=(False, 'Simulated export failure')):
            self.assertIsNone(core.combine_audio_chapters('incremental', chapter_index=0))
        self.assertEqual(self.first.read_bytes(), first_data)
        self.assertEqual(self.first.with_suffix('.vtt').read_bytes(), subtitle_data)
        self.assertEqual((self.root / 'chapter_export_1.json').read_bytes(), manifest)
        self.assertEqual(list(self.output.glob('.*')), [])
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
        self.first.with_suffix('.vtt').unlink()
        self.assertEqual(core.combine_audio_chapters('incremental', chapter_index=0), [str(self.first)])
        self.assertTrue(self.first.with_suffix('.vtt').exists())


if __name__ == '__main__':
    unittest.main()
