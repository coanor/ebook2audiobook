import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bs4 import BeautifulSoup
from ebooklib import epub

from lib.classes.book_chapters import chapter_documents, chapter_groups, chapter_catalog, select_chapter


def document(book, name, content):
    item = epub.EpubHtml(uid=name, file_name=name, title=name)
    item.content = content
    book.add_item(item)
    return item


def block(identifier, text, chapter, keep=True):
    return {'id': identifier, 'text': text, 'sentences': [text], 'keep': keep,
            'book_chapter': chapter}


class ChapterMappingTests(unittest.TestCase):
    def test_stale_parent_anchor_uses_its_heading_before_nested_chapter(self):
        book = epub.EpubBook()
        doc = document(book, 'part.xhtml', '<h1>第一篇  阅读的层次</h1>'
                       '<h2 id="one">第一章</h2><p>Chapter body.</p>')
        book.spine = [doc]
        book.toc = [(epub.Section('第一篇 阅读的层次', 'part.xhtml#missing'),
                     [epub.Link('part.xhtml#one', '第一章', 'one')])]
        items = list(chapter_documents(book, include_nested=True))
        items = [(doc, chapter) for doc, chapter in items
                 if BeautifulSoup(doc.get_content(), 'html.parser').body.get_text(strip=True)]
        self.assertEqual([chapter['title'] for _, chapter in items],
                         ['第一篇 阅读的层次', '第一章'])
        self.assertNotIn('Chapter body.', BeautifulSoup(items[0][0].get_content(), 'html.parser').get_text())
        self.assertIn('Chapter body.', BeautifulSoup(items[1][0].get_content(), 'html.parser').get_text())

    def test_nested_selection_keeps_its_subsections_and_excludes_sibling_chapters(self):
        book = epub.EpubBook()
        doc = document(book, 'part.xhtml', '<h1 id="part">上篇</h1>'
                       '<h2 id="one">第一章</h2><p>First chapter.</p>'
                       '<h3 id="section">第一节</h3><p>First subsection.</p>'
                       '<h2 id="two">第二章</h2><p>Second chapter.</p>')
        book.spine = [doc]
        book.toc = [(epub.Section('上篇', 'part.xhtml#part'), [
            (epub.Section('第一章', 'part.xhtml#one'),
             [epub.Link('part.xhtml#section', '第一节', 'section')]),
            epub.Link('part.xhtml#two', '第二章', 'two'),
        ])]
        items = list(chapter_documents(book, include_nested=True))
        catalog = chapter_catalog(items)
        self.assertEqual([chapter['title'] for chapter in catalog],
                         ['上篇', '第一章', '第一节', '第二章'])
        selected = select_chapter(catalog, '第一章')
        text = ' '.join(BeautifulSoup(doc.get_content(), 'html.parser').get_text()
                        for doc, chapter in items
                        if chapter['key'] == selected['key']
                        or selected['key'] in chapter.get('ancestors', []))
        self.assertIn('First chapter.', text)
        self.assertIn('First subsection.', text)
        self.assertNotIn('Second chapter.', text)
        self.assertEqual(len(chapter_catalog(chapter_documents(book))), 1)

    def test_get_blocks_selects_nested_chapter_and_groups_its_subsections(self):
        import zipfile
        import lib.core as core

        book = epub.EpubBook()
        part = document(book, 'part.xhtml', '<h1>上篇</h1>')
        one = document(book, 'one.xhtml', '<h2>第一章</h2>')
        section = document(book, 'section.xhtml', '<h3>第一节</h3>')
        two = document(book, 'two.xhtml', '<h2>第二章</h2>')
        book.spine = [(item.id, 'yes') for item in (part, one, section, two)]
        book.toc = [(epub.Section('上篇', 'part.xhtml'), [
            (epub.Section('第一章', 'one.xhtml'),
             [epub.Link('section.xhtml', '第一节', 'section')]),
            epub.Link('two.xhtml', '第二章', 'two'),
        ])]
        selected = {'key': 'one.xhtml#', 'title': '第一章', 'number': 2, 'occurrence': 1}
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'book.epub'
            with zipfile.ZipFile(archive, 'w'):
                pass
            session = dict(id='nested-test', cancellation_requested=False, language='zho',
                           language_iso1='zh', tts_engine='xtts', output_split=True,
                           output_split_hours='chapters', chapter_selection=selected,
                           epub_path=str(archive))
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)), \
                    patch.object(core, 'get_ebook_title', return_value='Book'), \
                    patch.object(core, 'filter_blocks', side_effect=lambda _, i, doc, *args: doc.file_name):
                blocks = core.get_blocks('nested-test', book)
            self.assertEqual(blocks, ['one.xhtml', 'section.xhtml'])
            self.assertEqual([chapter['title'] for chapter in session['book_chapter_labels']],
                             ['第一章', '第一章'])

    def test_book_title_wrapper_exposes_real_chapters_without_expanding_subsections(self):
        book = epub.EpubBook()
        book.set_title('旧唐书')
        intro = document(book, 'intro.xhtml', '<h1>旧唐书</h1>')
        first = document(book, 'first.xhtml', '<h1>本纪第一 高祖</h1>')
        body = document(book, 'body.xhtml', '<h2>武德元年</h2><p>正文</p>')
        second = document(book, 'second.xhtml', '<h1>本纪第二 太宗上</h1>')
        book.spine = [intro, first, body, second]
        book.toc = [(epub.Section('旧唐书', 'intro.xhtml'), [
            (epub.Section('本纪第一 高祖', 'first.xhtml'),
             [epub.Link('body.xhtml', '武德元年', 'sub')]),
            epub.Link('second.xhtml', '本纪第二 太宗上', 'second'),
        ])]
        items = list(chapter_documents(book))
        catalog = chapter_catalog(items)
        self.assertEqual([chapter['title'] for chapter in catalog],
                         ['旧唐书', '本纪第一 高祖', '本纪第二 太宗上'])
        selected = select_chapter(catalog, '高祖')
        self.assertEqual(selected['number'], 2)
        self.assertEqual([doc.file_name for doc, chapter in items if chapter['key'] == selected['key']],
                         ['first.xhtml', 'body.xhtml'])
        self.assertEqual(select_chapter(catalog, '2'), selected)

    def test_catalog_excludes_image_cover_and_typed_table_of_contents(self):
        book = epub.EpubBook()
        cover = document(book, 'cover.xhtml', '<img src="cover.jpg"/>')
        toc = document(book, 'toc.xhtml', '<section epub:type="toc"><p>Contents</p></section>')
        first = document(book, 'first.xhtml', '<h1>First</h1>')
        book.spine = [cover, toc, first]
        book.toc = [epub.Link('cover.xhtml', 'Cover', 'cover'),
                    epub.Link('toc.xhtml', 'Contents', 'toc'),
                    epub.Link('first.xhtml', 'First', 'first')]
        catalog = chapter_catalog(chapter_documents(book))
        self.assertEqual([chapter['title'] for chapter in catalog], ['First'])
        self.assertEqual(select_chapter(catalog, '1')['title'], 'First')

    def test_selection_rejects_missing_out_of_range_and_ambiguous_chapters(self):
        catalog = [{'key': 'a', 'number': 1, 'title': '第一章 上'},
                   {'key': 'b', 'number': 2, 'title': '第一章 下'}]
        for selector in ('0', '3', '', 'missing'):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                select_chapter(catalog, selector)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_chapter(catalog, '第一章')
        self.assertEqual(select_chapter(catalog, ' 第一章   下 ')['key'], 'b')

    def test_numeric_selection_of_duplicate_titles_survives_normalized_document_names(self):
        book = epub.EpubBook()
        first = document(book, 'first.xhtml', '<h1>列传第一百五十</h1><p>First</p>')
        second = document(book, 'second.xhtml', '<h1>列传第一百五十</h1><p>Second</p>')
        book.spine = [first, second]
        book.toc = [epub.Link('first.xhtml', '列传第一百五十', 'first'),
                    epub.Link('second.xhtml', '列传第一百五十', 'second')]
        catalog = chapter_catalog(chapter_documents(book))
        selected = select_chapter(catalog, '2')
        self.assertEqual(select_chapter(catalog, selected)['key'], 'second.xhtml#')
        renamed = [dict(chapter, key=f"normalized/{chapter['key']}") for chapter in catalog]
        self.assertEqual(select_chapter(renamed, selected)['key'], 'normalized/second.xhtml#')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_chapter(catalog, '列传第一百五十')

    def test_get_blocks_extracts_only_selected_chapter_documents(self):
        import zipfile
        import lib.core as core

        book = epub.EpubBook()
        first = document(book, 'first.xhtml', '<h1>First</h1>')
        second = document(book, 'second.xhtml', '<h1>Second</h1>')
        body = document(book, 'body.xhtml', '<p>Second body</p>')
        book.spine = [(item.id, 'yes') for item in (first, second, body)]
        book.toc = [epub.Link('first.xhtml', 'First', 'first'),
                    epub.Link('second.xhtml', 'Second', 'second')]
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'book.epub'
            with zipfile.ZipFile(archive, 'w'):
                pass
            selected = select_chapter(chapter_catalog(chapter_documents(book)), 'Second')
            session = dict(id='parse-test', cancellation_requested=False, language='zho',
                           language_iso1='zh', tts_engine='xtts', output_split=True,
                           output_split_hours='chapters', chapter_selection=selected,
                           epub_path=str(archive))
            def extract(_, index, doc, *args):
                return doc.file_name
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)), \
                    patch.object(core, 'get_ebook_title', return_value='Book'), \
                    patch.object(core, 'filter_blocks', side_effect=extract):
                blocks = core.get_blocks('parse-test', book)
            self.assertEqual(blocks, ['second.xhtml', 'body.xhtml'])
            self.assertEqual([label['title'] for label in session['book_chapter_labels']],
                             ['Second', 'Second'])

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
    def test_single_chapter_sample_exports_original_number_and_separate_filename(self):
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
            blocks = [block('a', 'Selected heading', {'key': 'selected', 'title': '第六章'}),
                      block('b', 'Selected body', {'key': 'selected', 'title': '第六章'})]
            for item in blocks:
                wave = np.sin(np.arange(12000) * 2 * np.pi * 220 / 24000).astype('float32') * 0.1
                sf.write(str(chapters / f"{item['id']}.flac"), wave, 24000)
                sentence_folder = sentences / item['id']
                sentence_folder.mkdir(parents=True)
                shutil.copyfile(chapters / f"{item['id']}.flac", sentence_folder / '0.flac')
            session = dict(id='sample-test', is_gui_process=False, cancellation_requested=False,
                           blocks_current={'blocks': blocks}, blocks_orig={'blocks': blocks},
                           process_dir=str(root), chapters_dir=str(chapters), sentences_dir=str(sentences),
                           audiobooks_dir=str(output), output_split=True, output_split_hours='chapters',
                           output_format='wav', output_channel='mono', cover=None,
                           metadata={'title': 'Book', 'creator': 'Author'}, final_name='book_sample_qwen3.wav',
                           chapter_selection={'number': 7, 'key': 'selected', 'title': '第六章'})
            full_output = output / 'book_qwen3_chapter7_第六章.wav'
            full_output.write_bytes(b'previous full-book output')
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)):
                files = core.combine_audio_chapters('sample-test')
            self.assertEqual([Path(file).name for file in files], ['book_sample_qwen3_chapter7_第六章.wav'])
            self.assertAlmostEqual(sf.info(files[0]).duration, 1.0, places=2)
            self.assertEqual(full_output.read_bytes(), b'previous full-book output')
            subtitles = Path(files[0]).with_suffix('.vtt').read_text()
            self.assertIn('Selected heading', subtitles)
            self.assertIn('Selected body', subtitles)

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
                           metadata={'title': 'Book', 'creator': 'Author'}, final_name='book_qwen3.wav')
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)):
                files = core.combine_audio_chapters('export-test')
            self.assertIsNotNone(files)
            self.assertEqual([Path(file).name for file in files],
                             ['book_qwen3_chapter1_第一章.wav', 'book_qwen3_chapter2_第二章.wav'])
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
