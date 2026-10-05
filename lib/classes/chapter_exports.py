"""Remember completed chapter exports across interrupted conversions."""

import hashlib
import json
import os
import tempfile
from multiprocessing.managers import DictProxy, ListProxy
from pathlib import Path


def _file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _plain(value):
    if isinstance(value, (dict, DictProxy)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, ListProxy)):
        return [_plain(item) for item in value]
    return value


class ChapterExportCache:
    def __init__(self, session, chapter_index, block_indices, final_file, audio_format):
        self.manifest = Path(session['process_dir']) / f'chapter_export_{chapter_index + 1}.json'
        self.outputs = [Path(final_file), Path(final_file).with_suffix('.vtt')]
        blocks = [session['blocks_current']['blocks'][i] for i in sorted(block_indices)]
        sources = []
        for block in blocks:
            sources.append(_file_stamp(Path(session['chapters_dir']) / f"{block['id']}.{audio_format}"))
            sentence_dir = Path(session['sentences_dir']) / block['id']
            sources.extend(_file_stamp(path) for path in sorted(sentence_dir.glob(f'*.{audio_format}')))
        # Editor state and chapter labels are not preserved in the SQLite
        # cache. Only sentence text and IDs affect the exported subtitles.
        content = [dict(id=block['id'], sentences=_plain(block['sentences'])) for block in blocks]
        cover = session['cover']
        cover_hash = (hashlib.sha256(Path(cover).read_bytes()).hexdigest()
                      if isinstance(cover, (str, os.PathLike)) else None)
        signature = dict(version=1, sources=sources, blocks=content,
                         metadata=_plain(session['metadata']), cover=cover_hash,
                         output_format=session['output_format'],
                         output_channel=session['output_channel'],
                         output=str(Path(final_file).resolve()))
        self.fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()

    def is_current(self):
        try:
            record = json.loads(self.manifest.read_text(encoding='utf-8'))
            return (record['fingerprint'] == self.fingerprint
                    and all(path.stat().st_size > 0 for path in self.outputs)
                    and record['outputs'] == [_file_stamp(path) for path in self.outputs])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def remember(self):
        record = dict(fingerprint=self.fingerprint,
                      outputs=[_file_stamp(path) for path in self.outputs])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.manifest.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, ensure_ascii=False)
            os.replace(temporary, self.manifest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
