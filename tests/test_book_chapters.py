import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bs4 import BeautifulSoup
from ebooklib import epub

from lib.classes.book_chapters import chapter_documents, chapter_groups


def document(book, name, content):
    item = epub.EpubHtml(uid=name, file_name=name, title=name)
    item.content = content
    book.add_item(item)
    return item


def block(identifier, text, chapter, keep=True):
    return {'id': identifier, 'text': text, 'sentences': [text], 'keep': keep,
            'book_chapter': chapter}


class ChapterMappingTests(unittest.TestCase):
    def test_parent_chapter_spans_files_and_subsections_in_spine_order(self):
        book = epub.EpubBook()
        second = document(book, 'a-second.xhtml', '<h1 id="second">Second</h1><p>End</p>')
        body = document(book, 'z-body.xhtml', '<h2 id="sub">Subsection</h2><p>Body</p>')
        first = document(book, 'z-first.xhtml', '<h1 id="first">First</h1>')
        book.spine = [first, body, second]
        book.toc = [(epub.Section('First', 'z-first.xhtml#first'),
                     [epub.Link('z-body.xhtml#sub', 'Subsection', 'sub')]),
                    epub.Link('a-second.xhtml#second', 'Second', 'second')]
        items = list(chapter_documents(book))
        self.assertEqual([doc.file_name for doc, _ in items],
                         ['z-first.xhtml', 'z-body.xhtml', 'a-second.xhtml'])
        original = [block(str(i), str(i), chapter) for i, (_, chapter) in enumerate(items)]
        self.assertEqual(chapter_groups(original, original),
                         [{'title': 'First', 'indices': [0, 1]}, {'title': 'Second', 'indices': [2]}])

    def test_two_main_chapters_in_one_document_preserve_all_text(self):
        book = epub.EpubBook()
        doc = document(book, 'text/章节.xhtml', '<p>Preface</p><section>'
                       '<h1 id="one">One</h1><p>A <b>bold</b> sentence.</p>'
                       '<h1 id="two">Two</h1><p>Final paragraph.</p></section>')
        book.spine = [doc]
        book.toc = [epub.Link('text/%E7%AB%A0%E8%8A%82.xhtml#one', 'One', 'one'),
                    epub.Link('text/章节.xhtml#two', 'Two', 'two')]
        items = list(chapter_documents(book))
        texts = [BeautifulSoup(item.get_content(), 'html.parser').body.get_text(' ', strip=True)
                 for item, _ in items]
        self.assertEqual(texts, ['Preface', 'One A bold sentence.', 'Two Final paragraph.'])
        self.assertEqual([chapter['title'] for _, chapter in items], ['Front matter', 'One', 'Two'])

    def test_book_without_toc_falls_back_to_spine_documents(self):
        book = epub.EpubBook()
        first = document(book, 'one.xhtml', '<h1>One</h1><p>Text</p>')
        second = document(book, 'two.xhtml', '<h1>Two</h1><p>Text</p>')
        book.spine = [first, second]
        self.assertEqual([chapter['key'] for _, chapter in chapter_documents(book)],
                         ['one.xhtml', 'two.xhtml'])

    def test_invalid_anchor_is_reported_instead_of_merging_chapters(self):
        book = epub.EpubBook()
        doc = document(book, 'one.xhtml', '<h1 id="one">One</h1><p>Text</p>')
        book.spine = [doc]
        book.toc = [epub.Link('one.xhtml#one', 'One', 'one'),
                    epub.Link('one.xhtml#missing', 'Two', 'two')]
        with self.assertRaisesRegex(ValueError, 'Cannot locate TOC chapter anchor'):
            list(chapter_documents(book))

    def test_edited_blocks_use_original_ids_and_skipped_blocks_are_excluded(self):
        original = [block('a', 'Original', {'key': 'first', 'title': 'First'}),
                    block('b', 'Skip', {'key': 'first', 'title': 'First'}),
                    block('c', 'End', {'key': 'second', 'title': 'Second'})]
        current = [block('a', 'Edited', None), block('b', 'Skip', None, keep=False),
                   block('c', 'End', None)]
        self.assertEqual(chapter_groups(current, original),
                         [{'title': 'First', 'indices': [0]}, {'title': 'Second', 'indices': [2]}])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required')
class ChapterExportTests(unittest.TestCase):
    def test_actual_wav_export_groups_blocks_and_limits_subtitles_to_each_chapter(self):
        import numpy as np
        import soundfile as sf
        import lib.core as core

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chapters = root / 'chapters'
            sentences = chapters / 'sentences'
            output = root / 'output'
            chapters.mkdir()
            output.mkdir()
            blocks = [block('a', 'First heading', {'key': 'first', 'title': '第一章'}),
                      block('b', 'First body', {'key': 'first', 'title': '第一章'}),
                      block('c', 'Second body', {'key': 'second', 'title': '第二章'})]
            for item in blocks:
                wave = np.sin(np.arange(12000) * 2 * np.pi * 220 / 24000).astype('float32') * 0.1
                sf.write(str(chapters / f"{item['id']}.flac"), wave, 24000)
                sentence_folder = sentences / item['id']
                sentence_folder.mkdir(parents=True)
                shutil.copyfile(chapters / f"{item['id']}.flac", sentence_folder / '0.flac')
            session = dict(id='export-test', is_gui_process=False, cancellation_requested=False,
                           blocks_current={'blocks': blocks}, blocks_orig={'blocks': blocks},
                           process_dir=str(root), chapters_dir=str(chapters), sentences_dir=str(sentences),
                           audiobooks_dir=str(output), output_split=True, output_split_hours='chapters',
                           output_format='wav', output_channel='mono', cover=None,
                           metadata={'title': 'Book', 'creator': 'Author'}, final_name='book.wav')
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)):
                files = core.combine_audio_chapters('export-test')
            self.assertIsNotNone(files)
            self.assertEqual([Path(file).name for file in files],
                             ['book_chapter1_第一章.wav', 'book_chapter2_第二章.wav'])
            self.assertAlmostEqual(sf.info(files[0]).duration, 1.0, places=2)
            self.assertAlmostEqual(sf.info(files[1]).duration, 0.5, places=2)
            first_vtt = Path(files[0]).with_suffix('.vtt').read_text()
            second_vtt = Path(files[1]).with_suffix('.vtt').read_text()
            self.assertIn('First heading', first_vtt)
            self.assertIn('First body', first_vtt)
            self.assertNotIn('Second body', first_vtt)
            self.assertIn('Second body', second_vtt)
            self.assertNotIn('First body', second_vtt)
            self.assertIn('00:00:00.000', second_vtt)


if __name__ == '__main__':
    unittest.main()
