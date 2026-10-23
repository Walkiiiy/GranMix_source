from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from config import ASSET_DIR


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(handle, value):
    handle.write(json.dumps(plain(value), allow_nan=False) + '\n')
    handle.flush()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


EXPECTED_VERSIONS = {'torch': '2.2.1', 'transformers': '4.45.2', 'numpy': '1.24.3'}


def runtime_versions():
    answer = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        answer[package] = {'actual': actual, 'historical': expected,
                           'matches': actual is not None and actual.split('+')[0] == expected}
    return answer


def check_assets(config, data_dir, tree_dir, model_dir, full_hash=False):
    identity = json.loads((ASSET_DIR / 'identity.json').read_text())
    rows = []
    for item in identity['files']:
        domain = item['path'].split('/')[0]
        if domain not in config['domains']:
            continue
        path = Path(data_dir) / item['path']
        ok = path.is_file() and path.stat().st_size == item['bytes']
        digest = sha256(path) if ok and full_hash else None
        if full_hash:
            ok = ok and digest == item['sha256']
        rows.append({'asset': 'data/' + item['path'], 'path': str(path), 'ok': ok, 'sha256': digest,
                     'check': 'size_and_sha256' if full_hash else 'size_only'})
    tree_identity_path = Path(tree_dir) / 'identity.json'
    try:
        tree_identity = json.loads(tree_identity_path.read_text())
    except (OSError, ValueError):
        tree_identity = {}
    if not isinstance(tree_identity, dict):
        tree_identity = {}
    tree_hashes = tree_identity.get('tree', {})
    if not isinstance(tree_hashes, dict):
        tree_hashes = {}
    rows.append({'asset': 'tree/identity.json', 'path': str(tree_identity_path),
                 'ok': tree_identity.get('version') == 1 and
                       set(tree_hashes) == {'nodes.json', 'train_leaf.npy'},
                 'sha256': sha256(tree_identity_path) if tree_identity_path.is_file() else None})
    for name in ('nodes.json', 'train_leaf.npy'):
        path = Path(tree_dir) / name
        digest = sha256(path) if path.is_file() else None
        rows.append({'asset': 'tree/' + name, 'path': str(path),
                     'ok': digest is not None and digest == tree_hashes.get(name), 'sha256': digest})
    tree_files = tree_identity.get('files', [])
    if not isinstance(tree_files, list):
        tree_files = []
    tree_sources = {item.get('path'): item for item in tree_files if isinstance(item, dict)}
    for expected in identity['files']:
        domain, filename = expected['path'].split('/')
        if domain in config['domains'] and filename == 'train.bin':
            source = tree_sources.get(expected['path'], {})
            rows.append({'asset': 'tree/source/' + expected['path'],
                         'path': str(tree_identity_path),
                         'ok': all(source.get(key) == expected[key] for key in ('bytes', 'sha256')),
                         'sha256': source.get('sha256'), 'check': 'training_corpus_identity'})
    path = Path(model_dir) / 'config.json'
    digest = sha256(path) if path.is_file() else None
    rows.append({'asset': 'model/config.json', 'path': str(path), 'ok': digest == identity['model_config_sha256'], 'sha256': digest})
    for domain in config['domains']:
        path = Path(data_dir) / domain / 'strat_labels.npy'
        rows.append({'asset': 'data/' + domain + '/strat_labels.npy', 'path': str(path), 'ok': not path.exists(), 'check': 'must_be_absent'})
    return {'ok': all(row['ok'] for row in rows), 'full_data_hash': full_hash,
            'checks': rows, 'runtime': runtime_versions()}


def asset_signature(report):
    """Content identity independent of the local mount path and runtime version."""
    if report is None:
        return None
    return {
        "full_data_hash": report["full_data_hash"],
        "checks": sorted([{k: v for k, v in row.items() if k != "path"}
                          for row in report["checks"]], key=lambda row: row["asset"]),
    }
