import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib import core


PREFACE = ('序言;《如何阅读一本书》的第一版是在 1940 年初出版的.'
           '很惊讶，我承认也很高兴的是，这本书立刻成为畅销书，高踞全美畅销书排行榜首有一年多时间.'
           '从 1940 年开始，这本书继续广泛的印刷发行，有精装本也有平装本，而且还被翻译成其他语言.'
           '法文、瑞典文、德文、西班牙文与意大利文.'
           '所以，为什么还要为目前这一代的读者再重新改写、编排呢？'
           '要这么做的原因，是近三十年来，我们的社会，与阅读这件事本身，都起了很大的变化。')


class ChineseSentenceTests(unittest.TestCase):
    def split(self, text, **settings):
        session = dict(language='zho', tts_engine='xtts', is_gui_process=False, **settings)
        with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)):
            result = core.get_sentences('test', text)
        self.assertIsNotNone(result)
        return result

    def assert_preserved(self, text, chunks):
        normalize = lambda value: ''.join(value.split())
        self.assertEqual(normalize(''.join(chunks)), normalize(text))
        for chunk in chunks:
            spoken = core.SML_TAG_PATTERN.sub('', chunk)
            self.assertTrue(any(char.isalnum() for char in spoken), chunk)
            self.assertLessEqual(len(spoken), core.language_mapping['zho']['max_chars'])

    def test_reported_preface_keeps_natural_sentence_boundaries(self):
        chunks = self.split(PREFACE)
        self.assert_preserved(PREFACE, chunks)
        for chunk in chunks[:-1]:
            self.assertIn(chunk[-1], '，。；！？、,.!?;:：', chunk)
        self.assertIn('从 1940 年开始', ''.join(chunks))
        self.assertFalse(any(chunk.endswith('从 1940 年') for chunk in chunks))

    def test_short_complete_sentence_is_not_cut_just_to_fill_a_buffer(self):
        text = '“很惊讶，我承认也很高兴的是，这本书立刻成为畅销书，高踞全美畅销书排行榜首有一年多时间。”'
        self.assertEqual(self.split(text), [text])

    def test_long_sentence_splits_at_clauses_and_retains_closing_quotes(self):
        text = '“' + '这是一个需要保持完整的长句子，' * 10 + '全文结束。”下一句话。'
        chunks = self.split(text)
        self.assert_preserved(text, chunks)
        self.assertGreater(len(chunks), 2)
        for chunk in chunks[:-1]:
            self.assertIn(chunk[-1], '，”', chunk)
        self.assertEqual(chunks[-1], '下一句话。')

    def test_mixed_language_numbers_and_sml_are_preserved(self):
        text = ('《测试》开始。[pause:1.4][voice:/tmp/reference.wav]'
                'OpenAI API version 2.5 costs 1,234.50 yuan。[/voice]'
                '下面是中文。[break]')
        chunks = self.split(text)
        self.assert_preserved(text, chunks)
        self.assertTrue(any('1,234.50' in chunk for chunk in chunks))
        self.assertTrue(any('2.5' in chunk for chunk in chunks))
        self.assertEqual(core.SML_TAG_PATTERN.findall(''.join(chunks)), core.SML_TAG_PATTERN.findall(text))

    def test_unpunctuated_text_stays_bounded_without_losing_text(self):
        text = '中华人民共和国历史研究以及社会经济发展的相关资料' * 12
        chunks = self.split(text)
        self.assert_preserved(text, chunks)
        self.assertGreater(len(chunks), 1)

    def test_long_unbroken_word_after_voice_tag_does_not_emit_control_only_chunk(self):
        text = '[voice:/tmp/reference.wav]' + 'A' * 200 + '。[/voice]'
        chunks = self.split(text)
        self.assert_preserved(text, chunks)

    def test_punctuation_at_length_limit_and_supplementary_chinese_are_preserved(self):
        for length in (80, 81, 82, 83, 160):
            with self.subTest(length=length):
                text = '甲' * length + '，”继续说话。' + '𠀀' * length + '。'
                chunks = self.split(text)
                self.assert_preserved(text, chunks)

    def test_translation_to_chinese_uses_chinese_splitting(self):
        with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: dict(
                language='eng', translate_enabled=True, translate='zho',
                tts_engine='xtts', is_gui_process=False))):
            chunks = core.get_sentences('translation', PREFACE)
        self.assert_preserved(PREFACE, chunks)
        self.assertTrue(all(chunk[-1] in '，。；！？、,.!?;:：' for chunk in chunks))

    def test_resume_replaces_old_chunks_and_resets_only_changed_current_sentence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = '从 1940 年开始，这本书继续广泛的印刷发行。'
            old_chunks = ['从 1940 年', '开始，这本书继续广泛的印刷发行。']
            block = dict(id='a', keep=True, text=text, sentences=old_chunks, voice=None)
            old = dict(blocks=[copy.deepcopy(block)], block_resume=0, sentence_resume=1)
            session = dict(id='resume', language='zho', tts_engine='xtts',
                           status=core.status_tags['CONVERTING'], cancellation_requested=False,
                           is_gui_process=False, ebook='book.epub', blocks_current=copy.deepcopy(old),
                           blocks_saved=copy.deepcopy(old), blocks_current_db=str(root / 'blocks.db'))
            saved_audio = root / 'a.flac'
            saved_audio.write_bytes(b'old audio remains until actual regeneration')
            with patch.object(core, 'context', SimpleNamespace(get_session=lambda _: session)), \
                    patch.object(core, 'convert_chapters2audio', return_value=False) as convert:
                _, success = core.finalize_audiobook('resume')
            self.assertFalse(success)
            convert.assert_called_once_with('resume')
            self.assertEqual(session['blocks_current']['blocks'][0]['sentences'], [text])
            self.assertEqual(session['blocks_current']['sentence_resume'], 0)
            self.assertEqual(session['blocks_saved']['blocks'][0]['sentences'], old_chunks)
            persisted = core.load_db_blocks(session['blocks_current_db'])
            self.assertEqual(persisted['blocks'][0]['sentences'], [text])
            self.assertEqual(persisted['sentence_resume'], 0)
            self.assertTrue(saved_audio.exists())


if __name__ == '__main__':
    unittest.main()
