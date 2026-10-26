"""Verify an extracted handover's SHA-256 manifest, without third-party packages."""
import hashlib
from pathlib import Path
import sys
import zipfile
from pathlib import PurePosixPath


def verify(root):
    if root.is_file() and root.suffix.lower() == '.zip':
        with zipfile.ZipFile(root) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP member names')
            listed = set()
            for line in archive.read('MANIFEST.sha256').decode('utf-8-sig').splitlines():
                if not line.strip():
                    continue
                expected, relative = line.split('  ', 1)
                path = PurePosixPath(relative)
                if path.is_absolute() or '..' in path.parts or '\\' in relative:
                    raise ValueError(f'Unsafe ZIP path: {relative}')
                actual = hashlib.sha256(archive.read(relative)).hexdigest()
                if actual.lower() != expected.lower():
                    raise ValueError(f'Checksum mismatch: {relative}')
                listed.add(relative)
            if set(names) != listed | {'MANIFEST.sha256'}:
                raise ValueError('ZIP has files absent from the manifest')
            print(f'Verified {len(listed)} payload files in {root.name}')
        return
    root = root.resolve()
    manifest = root / 'MANIFEST.sha256'
    checked = 0
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        expected, relative = line.split('  ', 1)
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Unsafe manifest path: {relative}')
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError(f'Checksum mismatch: {relative}')
        checked += 1
    print(f'Verified {checked} files in {root.name}')


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
