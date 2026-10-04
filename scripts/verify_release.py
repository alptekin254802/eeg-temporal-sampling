"""Verify snapshot identity without confusing it with numerical reproduction."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text('utf-8'))
    for relative,expected in manifest['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file(): raise ValueError('Missing or invalid path: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']: raise ValueError('Snapshot identity mismatch: '+relative)
    print(json.dumps(dict(snapshot_identity='PASS',files=len(manifest['files']),numerical_reproduction='Run scripts/verify_results.py separately'),indent=2))
if __name__=='__main__':main()
