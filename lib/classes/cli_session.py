"""Remember headless ebook sessions independently of shell scripts."""

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


def select_book_session(ebook, tmp_dir, process_name, language, translation=None,
                        explicit_session=None, new_session=False):
    """Return (session ID, reused), adopting matching older caches when needed.

    The source path identifies a book; the content checksum is used only when
    adopting an old session. Changes to the book are handled by core's existing
    checksum and block realignment logic.
    """
    source = Path(ebook).resolve()
    root = Path(tmp_dir)
    identity = {'ebook': str(source), 'language': language, 'translation': translation}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index_dir = root / 'cli_sessions'
    index_file = index_dir / f'{key}.json'
    session_id = explicit_session
    if session_id is not None:
        if not (root / f'proc-{session_id}').is_dir():
            raise ValueError('Session expired or does not exist!')
    elif not new_session:
        if index_file.exists():
            try:
                remembered = json.loads(index_file.read_text(encoding='utf-8'))['session_id']
                session_id = str(uuid.UUID(remembered))
            except (ValueError, KeyError, TypeError):
                session_id = None
            if session_id and not (root / f'proc-{session_id}').is_dir():
                session_id = None
        if session_id is None:
            # Pre-feature sessions have no index. Match the process directory
            # and original book checksum rather than guessing from a title.
            with source.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            process_key = hashlib.md5(process_name.encode()).hexdigest()
            candidates = []
            for checksum in root.glob(f'proc-*/{process_key}/checksum'):
                if checksum.read_text(encoding='utf-8').strip() == digest:
                    candidate = checksum.parent.parent.name.removeprefix('proc-')
                    try:
                        uuid.UUID(candidate)
                    except ValueError:
                        continue
                    candidates.append((checksum.parent.stat().st_mtime_ns, candidate))
            if candidates:
                session_id = max(candidates)[1]
    reused = session_id is not None
    session_id = session_id or str(uuid.uuid4())
    (root / f'proc-{session_id}').mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    # Write before conversion starts, and replace atomically so an interrupt
    # cannot leave a half-written pointer to the previous progress.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=index_dir,
                                         delete=False) as file:
            temporary = Path(file.name)
            json.dump(dict(identity, session_id=session_id), file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, index_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return session_id, reused
