import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib.external_tts import EXTERNAL_ENGINES, EngineWorker, engine_paths


class WorkerProtocolTests(unittest.TestCase):
    def worker(self, program):
        worker = EngineWorker([sys.executable, '-u', '-c', program])
        self.addCleanup(worker.close)
        return worker

    def test_multiple_unicode_requests_keep_one_process_and_close_it(self):
        worker = self.worker("import sys,json,os\nprint(json.dumps({'ok':True}))\n"
                             "for line in sys.stdin:\n r=json.loads(line); r.update(ok=True,pid=os.getpid()); print(json.dumps(r))")
        first = worker.synthesize('五千年，文明的起源。', '/tmp/章节.wav', None, 'Chinese', 'Uncle_Fu')
        second = worker.synthesize('下一句。', '/tmp/章节.wav', None, 'Chinese', 'Uncle_Fu')
        self.assertEqual(first['text'], '五千年，文明的起源。')
        self.assertEqual(first['output'], '/tmp/章节.wav')
        self.assertEqual(first['pid'], second['pid'])
        worker.close()
        self.assertIsNotNone(worker.process.poll())

    def test_initialization_failure_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing tokenizer'):
            self.worker("import json; print(json.dumps({'ok':False,'error':'Missing tokenizer'}))")

    def test_synthesis_failure_closes_worker(self):
        worker = self.worker("import json,sys\nprint(json.dumps({'ok':True}))\n"
                             "for line in sys.stdin: print(json.dumps({'ok':False,'error':'GPU out of memory'}))")
        with self.assertRaisesRegex(RuntimeError, 'GPU out of memory'):
            worker.synthesize('text', 'output', None, 'Chinese', 'Uncle_Fu')
        self.assertIsNotNone(worker.process.poll())

    def test_unexpected_exit_is_reported(self):
        worker = self.worker("import json,sys\nprint(json.dumps({'ok':True})); sys.stdin.readline()")
        with self.assertRaisesRegex(RuntimeError, 'exited unexpectedly'):
            worker.synthesize('text', 'output', None, 'Chinese', 'Uncle_Fu')


class ModelOutputTests(unittest.TestCase):
    def test_empty_index_generation_cannot_reuse_previous_sentence_audio(self):
        import numpy as np
        import soundfile as sf
        from lib.external_tts_worker import synthesize

        class EmptyModel:
            def infer(self, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'part.wav'
            sf.write(output, np.ones(100) * .1, 22050)
            request = dict(text='下一句。', voice='reference.wav', language='ZH', output=str(output))
            with self.assertRaises(RuntimeError):
                synthesize('indextts', EmptyModel(), request)
            self.assertFalse(output.exists())


class LauncherTests(unittest.TestCase):
    def test_all_external_engine_arguments_reach_the_existing_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy(Path(__file__).resolve().parents[1] / 'run.sh', root / 'run.sh')
            launcher = root / 'start-local.sh'
            launcher.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            launcher.chmod(0o755)
            book = root / '一本书.epub'
            book.touch()
            for engine in EXTERNAL_ENGINES:
                with self.subTest(engine=engine):
                    options = ['--speaker', 'Vivian'] if engine == 'qwen3' else []
                    completed = subprocess.run(['bash', str(root / 'run.sh'), str(book), engine, *options],
                                               capture_output=True, text=True, check=True)
                    args = json.loads(completed.stdout)
                    self.assertEqual(args[args.index('--tts_engine') + 1], engine)
                    self.assertEqual(args[args.index('--output_dir') + 1], str(root))
                    self.assertIn('--split_by_chapter', args)
                    self.assertEqual(args[args.index('--ebook') + 1], str(book))
                    if options:
                        self.assertEqual(args[-2:], options)


class ExternalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.process = self.root / 'process'
        self.process.mkdir()
        self.worker = patch.dict(os.environ, TXT2VOICE_RUNTIME_DIR=str(self.root))
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.requests = []

    def adapter(self, name, **options):
        from lib.classes.tts_engines.external import CosyVoice, Qwen3, IndexTTS
        import numpy as np
        import soundfile as sf
        paths = engine_paths(name, self.root)
        paths['python'].parent.mkdir(parents=True, exist_ok=True)
        paths['python'].touch()
        paths['source'].mkdir(parents=True, exist_ok=True)
        paths['reference'].parent.mkdir(parents=True, exist_ok=True)
        paths['reference'].touch()
        for filename in EXTERNAL_ENGINES[name]['required']:
            path = paths['model'] / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        class FakeWorker:
            closed = False

            def __init__(inner, *args, **kwargs):
                pass

            def synthesize(inner, text, output, voice, language, speaker):
                self.requests.append((text, voice, language, speaker))
                sf.write(output, np.ones(1600, dtype=np.float32) * .1, 16000)

            def close(inner):
                inner.closed = True

        session = dict(tts_engine=name, language='zho', device='cpu', custom_model=None,
                       fine_tuned='internal', model_cache=name, process_dir=str(self.process), voice=None)
        session.update(options)
        with patch('lib.classes.tts_engines.external.EngineWorker', FakeWorker):
            return {'cosyvoice': CosyVoice, 'qwen3': Qwen3, 'indextts': IndexTTS}[name](session)

    def test_all_engines_convert_into_sentence_cache_with_native_sample_rate(self):
        import soundfile as sf
        for name in EXTERNAL_ENGINES:
            with self.subTest(engine=name):
                adapter = self.adapter(name)
                output = self.process / f'{name}.flac'
                self.assertEqual(adapter.convert(output, '五千年，中华文明。'), (True, None))
                audio = sf.info(output)
                self.assertEqual(audio.samplerate, EXTERNAL_ENGINES[name]['samplerate'])
                self.assertAlmostEqual(audio.duration, .1, places=3)

    def test_pause_tags_stay_in_audio_and_are_not_sent_to_model(self):
        import soundfile as sf
        adapter = self.adapter('qwen3')
        output = self.process / 'sentence.flac'
        self.assertEqual(adapter.convert(output, '第一句。[pause:0.5]第二句。'), (True, None))
        self.assertEqual([request[0] for request in self.requests], ['第一句。', '第二句。'])
        self.assertAlmostEqual(sf.info(output).duration, .7, places=3)
        self.assertEqual(self.requests[0][3], 'Uncle_Fu')

    def test_failed_sentence_is_not_published_and_temporary_files_are_removed(self):
        adapter = self.adapter('qwen3')
        output = self.process / 'sentence.flac'
        with patch.object(adapter, 'audio_save', side_effect=RuntimeError('disk full')):
            success, error = adapter.convert(output, '测试句子。')
        self.assertFalse(success)
        self.assertIn('disk full', error)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.process.iterdir()), [])
        self.assertTrue(adapter.worker.closed)

    def test_qwen_reference_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot clone'):
            self.adapter('qwen3', voice='reference.wav')

    def test_qwen_speaker_is_case_insensitive(self):
        adapter = self.adapter('qwen3', tts_speaker='vivian')
        self.assertEqual(adapter.speaker, 'Vivian')


if __name__ == '__main__':
    unittest.main()
