"""Compile an optional KSP addon that enters only the OrbiterRL validation save.

Compilation writes to runs/bootstrap by default. --install additionally copies
the compiled addon into GameData/OrbiterRL; it refuses an existing destination.
The addon remains inactive unless KSP is launched with -orbiter-bootstrap.
Remove only GameData/OrbiterRL/OrbiterBootstrap.dll to uninstall it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ksp-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/bootstrap/OrbiterBootstrap.dll'))
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    managed = args.ksp_root / 'KSP.app/Contents/Resources/Data/Managed'
    if not managed.is_dir():
        managed = args.ksp_root / 'KSP_x64_Data/Managed'
    references = [managed / name for name in (
        'Assembly-CSharp.dll', 'UnityEngine.dll', 'UnityEngine.CoreModule.dll')]
    for reference in references:
        if not reference.is_file():
            parser.error(f'KSP assembly missing: {reference}')
    compiler = shutil.which('mcs')
    if compiler is None:
        parser.error('Mono C# compiler mcs is required')
    destination = args.ksp_root / 'GameData/OrbiterRL/OrbiterBootstrap.dll'
    if args.install and destination.exists():
        parser.error(f'Refusing to replace existing addon: {destination}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, '-target:library', f'-out:{args.output}',
                    *(f'-r:{reference}' for reference in references),
                    str(Path(__file__).with_name('ksp_bootstrap.cs'))], check=True)
    print(f'Compiled: {args.output.resolve()}')
    if args.install:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing another addon after compilation.
        with destination.open('xb') as target, args.output.open('rb') as source:
            shutil.copyfileobj(source, target)
        print(f'Installed: {destination}')
        print('Activate only on the intended run with KSP argument -orbiter-bootstrap')


if __name__ == '__main__':
    main()
