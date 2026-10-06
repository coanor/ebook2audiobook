"""Adapt isolated TTS workers to the existing sentence/chapter pipeline."""
import os
import tempfile
from pathlib import Path

from lib.classes.tts_engines.common.headers import TTSRegistry, TTSUtils, SML_TAG_PATTERN
from lib.external_tts import EXTERNAL_ENGINES, QWEN_SPEAKERS, EngineWorker, engine_paths


class ExternalTTS(TTSUtils):
    def __init__(self, session):
        self.session = session
        self.tts_engine = session['tts_engine']
        config = EXTERNAL_ENGINES[self.tts_engine]
        self.language = session.get('translate') if session.get('translate_enabled') else session['language']
        if self.language not in config['languages']:
            raise ValueError(f'Language {self.language} is not supported by {self.tts_engine}.')
        if session.get('custom_model') or session.get('fine_tuned', 'internal') != 'internal':
            raise ValueError('Use --tts_model_dir for these engines; only the downloaded model variants are supported.')
        device = session['device']
        if device not in ('cpu', 'cuda'):
            raise ValueError('The external engines currently support --device cpu or cuda.')
        self.paths = engine_paths(self.tts_engine, os.environ.get('TXT2VOICE_RUNTIME_DIR', os.getcwd()),
                                  session.get('tts_model_dir'))
        if not self.paths['python'].is_file() or not self.paths['source'].is_dir():
            setup = Path(__file__).resolve().parents[3] / 'setup-external-tts.sh'
            raise ValueError(f'Install {self.tts_engine} first: bash "{setup}" {self.tts_engine}')
        missing = [name for name in config['required'] if not (self.paths['model'] / name).is_file()]
        if missing:
            raise ValueError(f'Missing {self.tts_engine} model files under {self.paths["model"]}: {missing}')
        self.speaker = session.get('tts_speaker') or 'Uncle_Fu'
        if self.tts_engine == 'qwen3':
            supported = {name.lower(): name for name in QWEN_SPEAKERS}
            if self.speaker.lower() not in supported:
                raise ValueError(f'Unknown Qwen3 speaker {self.speaker}; choose one of {QWEN_SPEAKERS}.')
            self.speaker = supported[self.speaker.lower()]
        elif session.get('tts_speaker'):
            raise ValueError('--speaker is supported only by qwen3 CustomVoice.')
        self.tts_key = session['model_cache']
        self.params = {'samplerate': config['samplerate'], 'inline_voice': None}
        self.audio_segments = []
        _, error = self._set_voice(session.get('voice'))
        if error:
            raise ValueError(error)
        worker = Path(__file__).resolve().parents[2] / 'external_tts_worker.py'
        env = dict(os.environ)
        env['PATH'] = str(self.paths['python'].parent) + os.pathsep + env.get('PATH', '')
        self.worker = EngineWorker([str(self.paths['python']), '-u', str(worker),
                                    '--engine', self.tts_engine, '--source_dir', str(self.paths['source']),
                                    '--model_dir', str(self.paths['model']), '--device', device],
                                   cwd=str(self.paths['source']), env=env)

    def _set_voice(self, block_voice):
        if self.tts_engine == 'qwen3':
            if block_voice:
                return None, 'Qwen3 CustomVoice uses --speaker; it cannot clone a --voice reference recording.'
            return None, None
        voice = str(Path(block_voice).resolve()) if block_voice else str(self.paths['reference'])
        if not Path(voice).is_file():
            return None, f'Reference recording not found: {voice}. Pass --voice /path/reference.wav.'
        return voice, None

    def close(self):
        worker = getattr(self, 'worker', None)
        if worker:
            worker.close()

    def convert(self, sentence_file, sentence, **kwargs):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        temporary = None
        try:
            self.audio_segments = []
            self.params['block_voice'] = kwargs.get('block_voice', self.session.get('voice'))
            voice = self.params.get('inline_voice') or self.params['block_voice']
            self.params['current_voice'], error = self._set_voice(voice)
            if error:
                return False, error
            with tempfile.TemporaryDirectory(dir=self.session['process_dir'], prefix='external-tts-') as folder:
                for part in self._split_sentence_on_sml(sentence):
                    part = part.strip()
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if not success:
                            return False, error
                        continue
                    if not any(char.isalnum() for char in part):
                        continue
                    voice, error = self._set_voice(self.params['current_voice'])
                    if error:
                        return False, error
                    wav = Path(folder) / 'part.wav'
                    self.worker.synthesize(part, wav, voice,
                                           EXTERNAL_ENGINES[self.tts_engine]['languages'][self.language], self.speaker)
                    samples, rate = sf.read(wav, dtype='float32', always_2d=True)
                    if not samples.size or not np.isfinite(samples).all():
                        raise ValueError('TTS produced empty or invalid audio.')
                    audio = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
                    if rate != self.params['samplerate']:
                        audio = torchaudio.functional.resample(audio, rate, self.params['samplerate'])
                    self.audio_segments.append(audio)
            if not self.audio_segments:
                return False, 'TTS produced no audio for the sentence.'
            # A killed process must not leave a half-written sentence in the resume cache.
            with tempfile.NamedTemporaryFile(dir=Path(sentence_file).parent,
                                             suffix=Path(sentence_file).suffix, delete=False) as file:
                temporary = Path(file.name)
            self.audio_save(temporary, torch.cat(self.audio_segments, dim=-1), self.params['samplerate'])
            os.replace(temporary, sentence_file)
            return True, None
        except Exception as error:
            self.close()
            return False, self.log_exception(f'{self.tts_engine}.convert', error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class CosyVoice(ExternalTTS, TTSRegistry, name='cosyvoice'):
    pass


class Qwen3(ExternalTTS, TTSRegistry, name='qwen3'):
    pass


class IndexTTS(ExternalTTS, TTSRegistry, name='indextts'):
    pass
