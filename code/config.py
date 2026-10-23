from __future__ import annotations

import copy
import json
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
CONFIG_DIR = ROOT_DIR / 'configs'
ASSET_DIR = ROOT_DIR / 'assets'
CASES = ('ase', 'ghc4', 'bse', 'abse', 'ccghw', 'seven')


def load_config(case: str | None = None, path: str | Path | None = None) -> dict:
    if (case is None) == (path is None):
        raise ValueError('Specify exactly one of case or config path')
    if case is not None and case not in CASES:
        raise ValueError(f'Unknown case: {case}')
    config = json.loads((Path(path) if path else CONFIG_DIR / f'{case}.json').read_text())
    validate(config)
    return config


def validate(c: dict) -> None:
    if c.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema')
    domains = c['domains']
    if len(domains) < 2 or len(set(domains)) != len(domains):
        raise ValueError('At least two distinct ordered domains are required')
    tr, pr, dg = c['training'], c['probe'], c['diagnosis']
    for key in ('batch_size', 'sequence_length', 'max_steps', 'eval_batch_size'):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError(f'Invalid training {key}')
    if tr['sequence_length'] < 2 or not 0 <= c['seed'] < 2**32:
        raise ValueError('Invalid sequence length or seed')
    normal = pr['window_steps'] - len(domains) * pr['rounds'] * pr['steps']
    if normal <= 0 or normal != dg['window_steps'] or normal % dg['block_steps']:
        raise ValueError('Normal window, probe budget, and block size disagree')
    if not 0 < pr['one_hot_factor'] < 1:
        raise ValueError('This implementation requires mapped probes')
    if tr['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Unsupported model dtype')
    if c['target']['refresh_every_step'] is not True:
        raise ValueError('Main experiments refresh validation gradients before every measured step')
    if not 0 < c['target']['validation_fraction'] < 1:
        raise ValueError('Invalid target/development split')


def smoke_config() -> dict:
    """Configuration for a small synthetic run."""
    c = copy.deepcopy(load_config('ase'))
    c['name'] = 'synthetic_smoke'
    c.pop('historical')
    c['training'].update(batch_size=4, sequence_length=12, max_steps=24,
                         warmup_steps=2, eval_batch_size=4, dtype='float32')
    c['probe'].update(rounds=1, steps=1, window_steps=8, one_hot_factor=0.5)
    c['diagnosis'].update(window_steps=6, block_steps=2, history_blocks=9,
                          recent_blocks=3, history_decay=2**(-1/3), min_effective_count=1)
    c['control'].update(budget=4, delta=0.2)
    validate(c)
    return c
