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

    def test_batch_protocol_preserves_unicode_and_order(self):
        worker = self.worker("import sys,json\nprint(json.dumps({'ok':True}))\n"
                             "for line in sys.stdin:\n r=json.loads(line); r['ok']=True; print(json.dumps(r))")
        requests = [dict(text=text, output=f'/tmp/{i}.wav', language='Chinese', speaker='Uncle_Fu')
                    for i, text in enumerate(['第一句。', '第二句。'])]
        self.assertEqual(worker.synthesize_batch(requests)['batch'], requests)

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
    def test_cosyvoice_reuses_reference_features_and_refreshes_changed_recording(self):
        import numpy as np
        import torch
        from lib.external_tts_worker import synthesize
        from unittest.mock import Mock

        class FakeCosyVoice:
            sample_rate = 24000

            def __init__(self):
                self.add_zero_shot_spk = Mock(return_value=True)
                self.inference_cross_lingual = Mock(side_effect=lambda *args, **kwargs: iter([
                    {'tts_speech': torch.from_numpy(np.ones((1, 100), dtype='float32') * .1)}]))

        model = FakeCosyVoice()
        with tempfile.TemporaryDirectory() as folder:
            voice = Path(folder) / 'reference.wav'
            voice.write_bytes(b'reference')
            request = dict(text='测试。', voice=str(voice), output=str(Path(folder) / 'sentence.wav'))
            for _ in range(2):
                self.assertEqual(synthesize('cosyvoice', model, request), 24000)
            model.add_zero_shot_spk.assert_called_once()
            speaker_id = model.add_zero_shot_spk.call_args.args[2]
            self.assertTrue(speaker_id)
            for call in model.inference_cross_lingual.call_args_list:
                self.assertEqual(call.kwargs['zero_shot_spk_id'], speaker_id)
            voice.write_bytes(b'updated reference')
            synthesize('cosyvoice', model, request)
            self.assertEqual(model.add_zero_shot_spk.call_count, 2)
            other = Path(folder) / 'other.wav'
            other.write_bytes(b'other voice')
            request['voice'] = str(other)
            synthesize('cosyvoice', model, request)
            self.assertEqual(model.add_zero_shot_spk.call_count, 3)

    def test_qwen_model_receives_lists_and_writes_distinct_sentence_audio(self):
        import numpy as np
        import soundfile as sf
        from lib.external_tts_worker import synthesize_batch
        from unittest.mock import Mock

        model = Mock()
        model.generate_custom_voice.return_value = ([np.ones(100) * .1, np.ones(200) * .2], 24000)
        with tempfile.TemporaryDirectory() as folder:
            requests = [dict(text=text, output=str(Path(folder) / f'{i}.wav'),
                             language='Chinese', speaker='Uncle_Fu')
                        for i, text in enumerate(['第一句。', '第二句。'])]
            self.assertEqual(synthesize_batch('qwen3', model, requests), 24000)
            model.generate_custom_voice.assert_called_once_with(
                text=['第一句。', '第二句。'], language=['Chinese', 'Chinese'], speaker=['Uncle_Fu', 'Uncle_Fu'])
            self.assertEqual([sf.info(r['output']).frames for r in requests], [100, 200])
            # A malformed response cannot leave stale or partially published batch WAVs.
            for wavs in ([np.ones(10)], [np.ones(10), np.array([np.nan])]):
                model.generate_custom_voice.return_value = (wavs, 24000)
                with self.assertRaises(RuntimeError):
                    synthesize_batch('qwen3', model, requests)
                self.assertFalse(any(Path(r['output']).exists() for r in requests))

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
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        shutil.copy(Path(__file__).resolve().parents[1] / 'run.sh', self.root / 'run.sh')
        launcher = self.root / 'start-local.sh'
        launcher.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        launcher.chmod(0o755)
        self.book = self.root / '书本目录' / '一本书.epub'
        self.book.parent.mkdir()
        self.book.touch()
        self.env = dict(os.environ)
        self.env.pop('OUTPUT_DIR', None)

    def run_launcher(self, *options, check=True):
        return subprocess.run(['bash', str(self.root / 'run.sh'), *map(str, options)],
                              cwd=self.root, env=self.env, capture_output=True, text=True, check=check)

    def test_all_external_engine_arguments_reach_the_existing_pipeline(self):
        for engine in EXTERNAL_ENGINES:
            with self.subTest(engine=engine):
                options = ['--speaker', 'Vivian'] if engine == 'qwen3' else []
                args = json.loads(self.run_launcher(self.book, engine, *options).stdout)
                self.assertEqual(args[args.index('--tts_engine') + 1], engine)
                self.assertEqual(args[args.index('--output_dir') + 1], str(self.book.parent))
                self.assertIn('--split_by_chapter', args)
                self.assertEqual(args[args.index('--ebook') + 1], str(self.book))
                if options:
                    self.assertEqual(args[-2:], options)

    def test_named_input_and_engine_flags_accept_equals_spaces_and_any_order(self):
        cases = [('--tts=qwen3', '--ebook', self.book),
                 ('--ebook='+str(self.book), '--tts', 'QWEN3'),
                 ('--chapter', '第四章', '--tts=qwen3', '--ebook', self.book),
                 (self.book, '--tts_engine=qwen3'),
                 ('--tts_engine', 'qwen3', '--ebook', self.book)]
        for options in cases:
            with self.subTest(options=options):
                args = json.loads(self.run_launcher(*options).stdout)
                self.assertEqual(args[args.index('--tts_engine') + 1], 'qwen3')
                self.assertEqual(args.count('--tts_engine'), 1)
                self.assertEqual(args[args.index('--ebook') + 1], str(self.book))
                self.assertEqual(args.count('--ebook'), 1)
                self.assertEqual(args[args.index('--output_dir') + 1], str(self.book.parent))
                if '--chapter' in options:
                    self.assertEqual(args[-2:], ['--chapter', '第四章'])

    def test_raw_text_keeps_unicode_newlines_quotes_and_defaults_to_caller_directory(self):
        text = '五千年，文明的起源。\n他说："你好"。$HOME `literal`'
        for options in [('--text', text, '--tts=cosyvoice'), ('--tts=cosyvoice', '--text='+text)]:
            with self.subTest(options=options):
                args = json.loads(self.run_launcher(*options).stdout)
                self.assertEqual(args[args.index('--text') + 1], text)
                self.assertNotIn('--ebook', args)
                self.assertNotIn('--split_by_chapter', args)
                self.assertEqual(args[args.index('--output_dir') + 1], str(self.root))

    def test_output_flag_overrides_environment_and_resolves_relative_to_caller(self):
        self.env['OUTPUT_DIR'] = str(self.root / 'environment-output')
        for flag in [('--output_dir', 'custom output'), ('--output_dir=custom output',)]:
            args = json.loads(self.run_launcher('--tts=bark', '--ebook', self.book, *flag).stdout)
            self.assertEqual(args[args.index('--output_dir') + 1], str(self.root / 'custom output'))
            self.assertEqual(args.count('--output_dir'), 1)
        self.assertFalse((self.root / 'environment-output').exists())

    def test_text_starting_with_option_characters_stays_an_input_value(self):
        args = json.loads(self.run_launcher('--tts=cosyvoice', '--text=--tts=qwen3').stdout)
        self.assertIn('--text=--tts=qwen3', args)
        self.assertEqual(args[args.index('--tts_engine') + 1], 'cosyvoice')

    def test_application_accepts_engine_alias_and_equals_input_without_synthesis(self):
        app = Path(__file__).resolve().parents[1] / 'app.py'
        result = subprocess.run([sys.executable, str(app), '--tts=qwen3',
                                 '--text=五千年的文明。', '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--tts {', result.stdout)
        invalid = subprocess.run([sys.executable, str(app), '--tts=unknown', '--help'],
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('invalid choice', invalid.stderr)

    def test_batch_size_is_forwarded_and_invalid_sizes_are_rejected(self):
        args = json.loads(self.run_launcher('--tts=qwen3', '--text', '第一句。第二句。', '--batch_size', '2').stdout)
        self.assertEqual(args[args.index('--batch_size') + 1], '2')
        app = Path(__file__).resolve().parents[1] / 'app.py'
        for size in ('0', '17', 'four'):
            result = subprocess.run([sys.executable, str(app), '--tts=qwen3',
                                     '--batch_size', size, '--help'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, str(app), '--tts=cosyvoice', '--batch_size', '2',
                                 '--headless', '--text', '测试。'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--batch_size is supported only by qwen3', result.stderr)

    def test_invalid_inputs_fail_before_launcher_is_called(self):
        for options in [(), ('--tts=qwen3',), ('--ebook',), ('--text',),
                        ('--tts',), ('--ebook', self.book, '--tts'),
                        ('--text=',), ('--ebook', 'missing.epub'),
                        ('--ebook', self.book, '--text', 'text'),
                        ('--text', 'text', '--tts=unknown'),
                        ('--ebook', self.book, '--output_dir=')]:
            with self.subTest(options=options):
                result = self.run_launcher(*options, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('--headless', result.stdout)

    def test_help_exits_without_a_book(self):
        result = self.run_launcher('--tts=qwen3', '--help')
        self.assertIn('--ebook', result.stdout)
        self.assertIn('--text', result.stdout)
        self.assertIn('--tts=ENGINE', result.stdout)

    def test_omitting_engine_keeps_xtts_default(self):
        args = json.loads(self.run_launcher('--ebook', self.book).stdout)
        self.assertEqual(args[args.index('--tts_engine') + 1], 'xtts')


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

            def synthesize_batch(inner, requests):
                for request in requests:
                    inner.synthesize(request['text'], request['output'], None,
                                     request['language'], request['speaker'])

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

    def test_batch_preserves_pauses_order_and_sentence_cache(self):
        import soundfile as sf
        adapter = self.adapter('qwen3', tts_batch_size=4)
        items = [(self.process / '0.flac', '第一句。[pause:0.5]第二句。'),
                 (self.process / '1.flac', '[pause:0.2]'),
                 (self.process / '2.flac', '第三句。')]
        with patch.object(adapter.worker, 'synthesize_batch', wraps=adapter.worker.synthesize_batch) as batched:
            self.assertEqual(adapter.convert_batch(items), (True, None))
            batched.assert_called_once()
        self.assertEqual([r[0] for r in self.requests], ['第一句。', '第二句。', '第三句。'])
        for (path, _), duration in zip(items, [.7, .2, .1]):
            self.assertAlmostEqual(sf.info(path).duration, duration, places=3)
        self.assertEqual(sorted(p.name for p in self.process.iterdir()), ['0.flac', '1.flac', '2.flac'])

    def test_batch_worker_failure_does_not_publish_sentence_cache(self):
        adapter = self.adapter('qwen3', tts_batch_size=4)
        with patch.object(adapter.worker, 'synthesize_batch', side_effect=RuntimeError('GPU out of memory')):
            success, error = adapter.convert_batch([(self.process / '0.flac', '第一句。'),
                                                    (self.process / '1.flac', '第二句。')])
        self.assertFalse(success)
        self.assertIn('GPU out of memory', error)
        self.assertEqual(list(self.process.iterdir()), [])
        self.assertTrue(adapter.worker.closed)

    def test_qwen_speaker_is_case_insensitive(self):
        adapter = self.adapter('qwen3', tts_speaker='vivian')
        self.assertEqual(adapter.speaker, 'Vivian')


if __name__ == '__main__':
    unittest.main()
