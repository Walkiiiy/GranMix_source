from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ASSET_DIR, CASES, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='GranMix')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Print an experiment configuration')
    inspect.add_argument('--case', choices=CASES, required=True)
    preflight = sub.add_parser('preflight', help='Check data and environment')
    preflight.add_argument('--case', choices=(*CASES, 'all'), required=True)
    preflight.add_argument('--data', type=Path, required=True)
    preflight.add_argument('--tree', type=Path, required=True)
    preflight.add_argument('--model', type=Path, default=ASSET_DIR / 'model160')
    preflight.add_argument('--full-hash', action='store_true')
    train = sub.add_parser('train', help='Train GranMix or Uniform')
    train.add_argument('--case', choices=CASES, required=True)
    train.add_argument('--arm', choices=('granmix', 'uniform'), default='granmix')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--tree', type=Path, required=True)
    train.add_argument('--model', type=Path, default=ASSET_DIR / 'model160')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--device', default='cuda')
    train.add_argument('--end', type=int, help='Stop after this many optimizer updates while retaining the full learning-rate schedule')
    train.add_argument('--resume', type=Path, help='Trusted checkpoint from this implementation')
    train.add_argument('--evaluate-test', action='store_true')
    train.add_argument('--skip-evaluation', action='store_true')
    train.add_argument('--allow-version-drift', action='store_true')
    evaluate = sub.add_parser('evaluate', help='Evaluate a saved checkpoint')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--split', choices=('dev', 'test'), required=True)
    evaluate.add_argument('--data', type=Path)
    evaluate.add_argument('--tree', type=Path)
    evaluate.add_argument('--model', type=Path)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--device', default='cuda')
    evaluate.add_argument('--allow-version-drift', action='store_true')
    compare = sub.add_parser('compare', help='Compare paired Test results')
    compare.add_argument('--granmix', type=Path, required=True)
    compare.add_argument('--uniform', type=Path, required=True)
    smoke = sub.add_parser('smoke', help='Run a small synthetic CPU experiment')
    smoke.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        result = load_config(args.case)
    elif args.command == 'preflight':
        from utils import check_assets
        names = CASES if args.case == 'all' else (args.case,)
        result = {name: check_assets(load_config(name), args.data, args.tree, args.model, args.full_hash)
                  for name in names}
        print(json.dumps(result, indent=2))
        return 0 if all(r['ok'] and all(v['matches'] for v in r['runtime'].values())
                        for r in result.values()) else 1
    elif args.command == 'train':
        from train import train as run_training
        result = run_training(load_config(args.case), arm=args.arm, data_dir=args.data,
            tree_dir=args.tree, model_dir=args.model, output_dir=args.output, device=args.device,
            end=args.end, resume=args.resume, evaluate_test=args.evaluate_test,
            skip_evaluation=args.skip_evaluation, allow_version_drift=args.allow_version_drift)
    elif args.command == 'evaluate':
        result = _evaluate_checkpoint(args)
    elif args.command == 'compare':
        from train import compare_runs
        result = compare_runs(args.granmix, args.uniform)
    else:
        import torch
        from config import smoke_config
        from data import create_fixture
        from train import create_smoke_model, train as run_training
        torch.set_num_threads(2)
        args.output.mkdir(parents=True, exist_ok=False)
        data, tree = create_fixture(args.output / 'fixture')
        result = {}
        for arm in ('granmix', 'uniform'):
            result[arm] = run_training(smoke_config(), arm=arm, data_dir=data, tree_dir=tree,
                model_dir=ASSET_DIR / 'model160', output_dir=args.output / arm, device='cpu',
                evaluate_test=True, allow_version_drift=True, check_identity=False,
                model_factory=create_smoke_model)
    print(json.dumps(result, indent=2))
    return 0


def _evaluate_checkpoint(args):
    from data import TreeCorpus
    from train import create_model, evaluate, load_checkpoint
    from utils import check_assets, runtime_versions, write_json
    if args.output.exists():
        raise FileExistsError('Evaluation output already exists')
    saved = load_checkpoint(args.checkpoint)
    manifest, config = saved['manifest'], saved['config']
    versions = runtime_versions()
    if not args.allow_version_drift and not all(v['matches'] for v in versions.values()):
        raise ValueError('Runtime version mismatch; use --allow-version-drift to continue')
    data = args.data or Path(manifest['data_dir'])
    tree = args.tree or Path(manifest['tree_dir'])
    model_dir = args.model or Path(manifest['model_dir'])
    checks = check_assets(config, data, tree, model_dir, full_hash=True)
    if not checks['ok']:
        raise ValueError('Data, model, or tree identity mismatch')
    model = create_model(model_dir, config['training'], args.device)
    model.load_state_dict(saved['model'], strict=True)
    corpus = TreeCorpus(tree, data, config['domains'], config['training']['sequence_length'], config['seed'])
    result = evaluate(model, corpus, split=args.split, pad_id=config['training']['pad_token_id'],
        batch_size=config['training']['eval_batch_size'], validation_fraction=config['target']['validation_fraction'])
    result.update(checkpoint=str(args.checkpoint.resolve()), nominal_step=saved['clock']['nominal_step'],
                  actual_step=saved['clock']['actual_step'], runtime=versions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
