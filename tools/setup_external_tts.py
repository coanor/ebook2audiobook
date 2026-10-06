"""Install isolated inference environments without starting TTS or a web server."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'lib'))
from external_tts import EXTERNAL_ENGINES, engine_paths


def run(command, **kwargs):
    env = dict(os.environ)
    env['UV_BUILD_CONSTRAINT'] = str(REPO / 'tools' / 'external-tts-build.txt')
    env['UV_CONSTRAINT'] = str(REPO / 'tools' / 'external-tts-runtime.txt')
    kwargs.setdefault('env', env)
    subprocess.run([os.fspath(value) for value in command], check=True, **kwargs)


def prepare_sox(paths, runtime):
    if shutil.which('sox'):
        return
    bundle = runtime / 'run' / 'external-tts' / 'system'
    required = ('usr/bin/sox', 'usr/lib/x86_64-linux-gnu/libsox.so.3',
                'usr/lib/x86_64-linux-gnu/libltdl.so.7')
    if not all((bundle / name).is_file() for name in required):
        if not shutil.which('apt') or not shutil.which('dpkg-deb'):
            raise SystemExit('Install the SoX command before setting up these TTS engines.')
        bundle.mkdir(parents=True, exist_ok=True)
        run(['apt', 'download', 'sox', 'libsox3', 'libltdl7'], cwd=bundle)
        for package in bundle.glob('*.deb'):
            run(['dpkg-deb', '--extract', package, bundle])
    wrapper = bundle / 'bin/sox'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        f'#!{sys.executable}\nimport os,sys\n'
        "existing = os.environ.get('LD_LIBRARY_PATH', '')\n"
        f"os.environ['LD_LIBRARY_PATH'] = {str(bundle / 'usr/lib/x86_64-linux-gnu')!r} + "
        "(os.pathsep + existing if existing else '')\n"
        f"os.execv({str(bundle / 'usr/bin/sox')!r}, ['sox', *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    run([wrapper, '--version'])
    alias = paths['python'].parent / 'sox'
    if not alias.exists():
        alias.symlink_to(wrapper)


def install(engine, runtime):
    paths = engine_paths(engine, runtime)
    if not paths['source'].is_dir():
        raise SystemExit(f'Source not downloaded: {paths["source"]}')
    print(f'Installing {engine} in {paths["python"].parent.parent}', flush=True)
    if not paths['python'].is_file():
        run(['uv', 'venv', '--python', '3.11', '--seed', paths['python'].parent.parent])
    prepare_sox(paths, runtime)
    run(['uv', 'pip', 'install', '--python', paths['python'], '--index-url',
         'https://download.pytorch.org/whl/cu128',
         'torch==2.8.0+cu128', 'torchaudio==2.8.0+cu128', 'torchvision==0.23.0+cu128'])
    if engine == 'cosyvoice':
        run(['uv', 'pip', 'install', '--python', paths['python'], '-r',
             REPO / 'tools' / 'cosyvoice-inference.txt'])
    elif engine == 'qwen3':
        run(['uv', 'pip', 'install', '--python', paths['python'], 'setuptools==80.9.0', '-e', paths['source']])
    else:
        # Upstream's OpenCV 4.9 wheel predates its pinned NumPy 2.x ABI.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as overrides:
            overrides.write('opencv-python==4.11.0.86\n')
            overrides.flush()
            run(['uv', 'pip', 'install', '--python', paths['python'],
                 '--overrides', overrides.name, 'setuptools==80.9.0', '-e', paths['source']])
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join((str(paths['python'].parent), str(Path.home() / '.local/bin'), env.get('PATH', '')))
    run([paths['python'], REPO / 'lib' / 'external_tts_worker.py', '--engine', engine,
         '--source_dir', paths['source'], '--model_dir', paths['model'], '--check-imports'], env=env)
    print(f'{engine}: imports and CUDA check passed; no model loaded or speech generated.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', nargs='?', choices=['all', *EXTERNAL_ENGINES], default='all')
    parser.add_argument('--runtime-dir', type=Path, required=True)
    args = parser.parse_args()
    if not shutil.which('uv'):
        raise SystemExit('uv is required; install it before running this setup script.')
    for engine in EXTERNAL_ENGINES if args.engine == 'all' else [args.engine]:
        install(engine, args.runtime_dir.resolve())


if __name__ == '__main__':
    main()
