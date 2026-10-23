from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from ghost import DomainGhost
from torch.nn import functional as F

from config import CODE_DIR, validate
from data import MappedProbeSampler, TreeCorpus, UniformSampler
from diagnosis import DiagnosisConfig, PartitionDiagnoser
from refinement import GranMixController
from utils import (
    append_json,
    asset_signature,
    check_assets,
    plain,
    runtime_versions,
    sha256,
    write_json,
)


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_dir, training: dict, device='cuda'):
    from transformers import AutoConfig, GPTNeoXForCausalLM
    dtype = getattr(torch, training['dtype'])
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True,
                                    trust_remote_code=False,
                                    attn_implementation=training['attention'], torch_dtype=dtype)
    if cfg.model_type != 'gpt_neox':
        raise ValueError('Expected a GPT-NeoX/Pythia configuration')
    # Initialize in the target dtype; casting from FP32 changes the sampled weights.
    model = GPTNeoXForCausalLM._from_config(cfg, torch_dtype=dtype)
    return model.to(device)


def model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def create_smoke_model(device='cpu'):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=32, rotary_pct=0.5,
                        hidden_dropout=0.0, attention_dropout=0.0,
                        bos_token_id=0, eos_token_id=0, use_cache=False)
    cfg._attn_implementation = 'eager'
    return GPTNeoXForCausalLM(cfg).to(device)


def token_losses(logits, input_ids, pad_id):
    """Shifted CE [B, L-1] in model dtype; training includes masked zero slots."""
    if input_ids.ndim != 2 or input_ids.shape[1] < 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("expected matching [batch, sequence>=2] token/logit dimensions")
    labels = input_ids[:, 1:].clone()
    if pad_id is not None:
        labels.masked_fill_(labels == pad_id, -100)
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                           labels.contiguous().view(-1), reduction="none", ignore_index=-100).view_as(labels)


def _logits(model, ids):
    output = model(input_ids=ids, use_cache=False)
    return output["logits"] if isinstance(output, dict) else output.logits


def training_loss(model, ids, pad_id):
    return token_losses(_logits(model, ids), ids, pad_id).mean()


def sequence_loss_totals(model, ids, pad_id):
    losses = token_losses(_logits(model, ids), ids, pad_id).to(device="cpu", dtype=torch.float64)
    valid = torch.ones_like(ids[:, 1:], dtype=torch.bool) if pad_id is None else ids[:, 1:] != pad_id
    return losses.sum(dim=1), valid.sum(dim=1).cpu()


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, initial_lr, final_lr):
        self.total_steps, self.warmup_steps = int(total_steps), int(warmup_steps)
        self.initial_lr, self.final_lr = float(initial_lr), float(final_lr)
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError('Warmup must be shorter than the scheduled horizon')
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step > self.total_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        if step < self.warmup_steps:
            return [self.initial_lr + (base - self.initial_lr) * step / self.warmup_steps
                    for base in self.base_lrs]
        return [self.final_lr + (base - self.final_lr) *
                (1 + math.cos(math.pi * (step - self.warmup_steps) /
                              (self.total_steps - self.warmup_steps))) / 2
                for base in self.base_lrs]


def create_optimizer(model, config):
    # Keep parameter order; exclude LayerNorm parameters and biases from weight decay.
    norm_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
                for p in module.parameters()}
    groups = [[], []]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[int('bias' in name or id(parameter) in norm_ids)].append(parameter)
    optimizer = torch.optim.AdamW(
        [{'params': groups[0], 'weight_decay': config['weight_decay']},
         {'params': groups[1], 'weight_decay': 0.0}],
        lr=config['learning_rate'], betas=tuple(config['adam_betas']), eps=config['adam_epsilon'])
    scheduler = WarmupCosine(optimizer, config['max_steps'], config['warmup_steps'],
                            config['initial_learning_rate'], config['final_learning_rate'])
    return optimizer, scheduler


def global_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_global_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG layout differs from the current host')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # RNG states require pickle loading; use trusted checkpoints.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'granmix-main-v1':
        raise ValueError('Unsupported checkpoint format')
    return payload


def evaluate(model, corpus, *, split, pad_id, batch_size=8, validation_fraction=0.8):
    if split not in ('dev', 'test'):
        raise ValueError('Only independent development and complete Test evaluation are exposed')
    original_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    device = next(model.parameters()).device
    domain_losses = {d: [] for d in corpus.domains}
    domain_tokens = {d: 0 for d in corpus.domains}
    pending_ids, pending_domains = [], []

    def flush():
        if not pending_ids:
            return
        ids = torch.from_numpy(np.stack(pending_ids)).to(device)
        with torch.no_grad():
            values, counts = sequence_loss_totals(model, ids, pad_id)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite held-out loss')
        for domain, value, count in zip(pending_domains, values, counts):
            domain_losses[domain].append(value.item())
            domain_tokens[domain] += int(count)
        pending_ids.clear()
        pending_domains.clear()

    try:
        # Batches may span domain boundaries.
        for domain in corpus.domains:
            source = 'val' if split == 'dev' else 'test'
            count = corpus.split_size(domain, source)
            start = int(count * validation_fraction) if split == 'dev' else 0
            for row in range(start, count):
                pending_ids.append(corpus.validation(domain, [row], split=source)[0])
                pending_domains.append(domain)
                if len(pending_ids) == batch_size:
                    flush()
        flush()
    finally:
        for module, training in original_modes:
            module.training = training
    if any(count == 0 for count in domain_tokens.values()):
        raise ValueError('Every domain must have held-out target tokens')
    per_domain = {name: math.fsum(domain_losses[name]) / domain_tokens[name]
                  for name in sorted(domain_losses)}
    mean = math.fsum(per_domain.values()) / len(per_domain)
    return {'split': split, 'complete_split': True, 'eval_limit': 0,
            'per_domain_nll': per_domain, 'mean_nll': mean, 'geometric_ppl': math.exp(mean),
            'sequences': {name: len(values) for name, values in domain_losses.items()},
            'tokens': domain_tokens,
            'aggregation': 'token CE sum / target tokens within domain -> equal domain NLL mean -> exp'}


def compare_runs(granmix_path, uniform_path):
    entries = []
    for filename, expected_arm in ((granmix_path, 'granmix'), (uniform_path, 'uniform')):
        path = Path(filename)
        summary = json.loads(path.read_text())
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if (summary.get('arm') != expected_arm or not summary.get('complete')
                or summary.get('synthetic_fixture') or manifest.get('partial_run')):
            raise ValueError('Only complete, nonsynthetic main runs can form a result pair')
        test = summary.get('evaluations', {}).get('test')
        if not test or not test.get('complete_split') or test.get('eval_limit') != 0:
            raise ValueError('Both runs need full Test evaluation')
        entries.append((summary, manifest, test))
    a, b = entries
    for key in ('name', 'domains', 'seed', 'training'):
        if a[1]['config'][key] != b[1]['config'][key]:
            raise ValueError(f'Mismatched paired condition: {key}')
    if a[1]['initial_model_sha256'] != b[1]['initial_model_sha256']:
        raise ValueError('Paired initial model weights differ')
    if asset_signature(a[1]['asset_identity']) != asset_signature(b[1]['asset_identity']) or a[1]['runtime'] != b[1]['runtime']:
        raise ValueError('Paired data identities or runtime differ')
    gm, uniform = a[2]['geometric_ppl'], b[2]['geometric_ppl']
    return {'case': a[1]['config']['name'], 'seed': a[1]['config']['seed'],
            'granmix_test_ppl': gm, 'uniform_test_ppl': uniform,
            'delta': gm - uniform, 'relative_change_percent': 100 * (gm / uniform - 1),
            'source': 'full Test evaluations'}


def loader_seed_draw():
    """Match DataLoader RNG draws per probe segment, normal window, or Uniform iterator."""
    torch.empty((), dtype=torch.int64).random_()


def _source_hashes():
    return {str(p.relative_to(CODE_DIR)): sha256(p) for p in sorted(CODE_DIR.rglob('*.py'))}


def train(config, *, arm, data_dir, tree_dir, model_dir, output_dir, device='cuda',
          end=None, resume=None, evaluate_test=False, skip_evaluation=False,
          allow_version_drift=False, check_identity=True, model_factory=None):
    """Train or resume for a fixed number of optimizer updates."""
    validate(config)
    if arm not in ('granmix', 'uniform'):
        raise ValueError('Supported arms are granmix and uniform')
    tr, pr = config['training'], config['probe']
    endpoint = tr['max_steps'] if end is None else int(end)
    if not 0 < endpoint <= tr['max_steps']:
        raise ValueError('Endpoint lies outside the original LR horizon')
    versions = runtime_versions()
    if not allow_version_drift and not all(v['matches'] for v in versions.values()):
        raise ValueError('Runtime version mismatch; run preflight or use --allow-version-drift.')
    asset_report = None
    if check_identity:
        asset_report = check_assets(config, data_dir, tree_dir, model_dir, full_hash=True)
        if not asset_report['ok']:
            failures = [r['path'] for r in asset_report['checks'] if not r['ok']]
            raise ValueError(f'Asset identity mismatch: {failures}')
    corpus = TreeCorpus(tree_dir, data_dir, config['domains'],
                        sequence_length=tr['sequence_length'], seed=config['seed'])
    seed_training(config['seed'])
    model = create_model(model_dir, tr, device) if model_factory is None else model_factory(device)
    optimizer, scheduler = create_optimizer(model, tr)
    initial_sha = model_fingerprint(model)
    expected_initial = config.get('historical', {}).get('initial_model_sha256')
    if check_identity and not allow_version_drift and initial_sha != expected_initial:
        raise ValueError('Initial model weights do not match the experiment configuration')
    diag = controller = ghost = None
    if arm == 'granmix':
        diag = PartitionDiagnoser(corpus.parents, corpus.reference_mass, DiagnosisConfig(**config['diagnosis']))
        controller = GranMixController(corpus.children, corpus.reference_mass, list(corpus.source_cut),
                                       **config['control'])
        ghost = DomainGhost(model, clip=config['diagnosis']['clip'],
                            norm_floor=config['diagnosis']['probe_norm_floor'])
        sampler = MappedProbeSampler(corpus, config['seed'])
    else:
        sampler = UniformSampler(corpus, config['seed'], batch_size=tr['batch_size'])
    target_rng = np.random.default_rng(config['target']['seed_offset'] + config['seed'])
    target_indices = {}
    for domain in corpus.domains:
        available = int(corpus.split_size(domain, 'val') * config['target']['validation_fraction'])
        target_indices[domain] = target_rng.choice(available,
            config['target']['sequences_per_domain'], replace=False).tolist()
    clock = {'nominal_step': 0, 'actual_step': 0, 'probe_updates': 0, 'normal_updates': 0, 'window': 0}
    progress = {'phase': 'start', 'order': [], 'target_index': 0,
                'probe_batch_index': 0, 'normal_done': 0, 'segment_sampler_state': None}
    if resume:
        restored = load_checkpoint(resume)
        if restored['config'] != config or restored['arm'] != arm:
            raise ValueError('Resume arm/config differs from the checkpoint')
        if restored['target_indices'] != target_indices:
            raise ValueError('Validation targets differ from the checkpoint')
        if asset_signature(restored['asset_identity']) != asset_signature(asset_report):
            raise ValueError('Resume asset contents differ from the checkpoint')
        if not allow_version_drift and restored['manifest']['runtime'] != versions:
            raise ValueError('Resume runtime differs from the checkpoint')
        model.load_state_dict(restored['model'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        corpus.load_state_dict(restored['corpus'])
        sampler.load_state_dict(restored['sampler'])
        clock.update(restored['clock'])
        if clock['actual_step'] >= endpoint:
            raise ValueError('Resume checkpoint must precede the requested endpoint')
        if clock['nominal_step'] != clock['actual_step']:
            raise ValueError('Resume checkpoint uses the old nominal update budget')
        if ghost:
            if 'progress' not in restored:
                raise ValueError('Resume checkpoint does not contain training phase state')
            progress = copy.deepcopy(restored['progress'])
            if progress['phase'] not in ('start', 'probe', 'normal'):
                raise ValueError('Invalid checkpoint training phase')
            if progress['normal_done'] != restored['diagnosis']['window_steps']:
                raise ValueError('Checkpoint training and diagnosis windows disagree')
            diag = PartitionDiagnoser.from_state_dict(restored['diagnosis'])
            controller.load_state_dict(restored['controller'])
            # Preserve validated weights exactly; normalization can move them by one ULP.
            saved_weights = np.array(restored['controller']['weights'], dtype=np.float64, copy=True)
            if not np.allclose(saved_weights, controller.weights, rtol=0, atol=1e-12):
                raise ValueError('Checkpoint weights require nontrivial renormalization')
            controller.weights = saved_weights
            ghost.restore_boundary_rng(restored['ghost_rng'])
        restore_global_rng(restored['global_rng'])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'format': 'granmix-main-v1', 'config': config, 'arm': arm,
                'source_sha256': _source_hashes(), 'runtime': versions,
                'asset_identity': asset_report, 'initial_model_sha256': initial_sha,
                'historical_initial_model_matches': initial_sha == expected_initial,
                'target_indices': target_indices, 'endpoint': endpoint,
                'partial_run': endpoint < tr['max_steps'], 'device': str(device),
                'model_dir': str(Path(model_dir).resolve()), 'data_dir': str(Path(data_dir).resolve()),
                'tree_dir': str(Path(tree_dir).resolve()), 'synthetic_fixture': model_factory is not None,
                'resume': None if resume is None else str(Path(resume).resolve()),
                'reported_metrics_are_new_results': True,
                'version_drift_permitted': allow_version_drift}
    write_json(out / 'manifest.json', manifest)
    model.zero_grad(set_to_none=True)
    started = time.monotonic()

    def target_loss(domain):
        ids = torch.from_numpy(corpus.validation(domain, target_indices[domain], split='val')).to(device)
        return training_loss(model, ids, tr['pad_token_id'])

    def refresh_targets():
        ghost.refresh({i: (lambda d=d: target_loss(d)) for i, d in enumerate(corpus.domains)})

    def checkpoint():
        name = f"step_{clock['actual_step']:06d}.pt"
        payload = {'format': 'granmix-main-v1', 'config': config, 'arm': arm,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'clock': copy.deepcopy(clock),
                   'progress': copy.deepcopy(progress) if ghost is not None else None,
                   'corpus': corpus.state_dict(), 'sampler': sampler.state_dict(),
                   'global_rng': global_rng_state(), 'target_indices': target_indices,
                   'asset_identity': asset_report,
                   'diagnosis': None if diag is None else diag.state_dict(),
                   'controller': None if controller is None else controller.state_dict(),
                   'ghost_rng': None if ghost is None else ghost.boundary_rng_state(),
                   'manifest': manifest}
        save_checkpoint(out / 'checkpoints' / name, payload)
        return name

    try:
        with ExitStack() as stack:
            logs = {name: stack.enter_context((out / (name + '.jsonl')).open('w'))
                    for name in ('training', 'observations', 'probes', 'boundaries')}

            def update(batch, phase, measured):
                model.train()
                ids = torch.from_numpy(batch['ids']).to(device)
                if measured:
                    ghost.begin(batch['domain_ids'], [tr['sequence_length'] - 1] * len(ids))
                loss = training_loss(model, ids, tr['pad_token_id'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                if measured:
                    readout = ghost.backward(loss)
                    contributions = np.asarray(readout.contributions, dtype=np.float64)
                    importance = batch['p_i'] / batch['q_i']
                    if phase == 'normal':
                        diag.observe_normal_batch(batch['leaf_nodes'], contributions, importance)
                    else:
                        diag.observe_probe_history(batch['leaf_nodes'], contributions, importance)
                    append_json(logs['observations'], dict(actual_step=clock['actual_step'] + 1,
                        phase=phase, window=clock['window'], leaf_nodes=batch['leaf_nodes'],
                        importance_weights=importance, contributions=contributions,
                        used_for_normal_audit=phase == 'normal'))
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr['max_grad_norm'])
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError('Nonfinite training gradient')
                lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                clock['actual_step'] += 1
                clock['nominal_step'] = clock['actual_step']
                clock['probe_updates' if phase == 'probe' else 'normal_updates'] += 1
                append_json(logs['training'], dict(actual_step=clock['actual_step'],
                    nominal_before=clock['actual_step'] - 1, phase=phase, window=clock['window'],
                    loss=float(loss.detach()), lr=lr,
                    batch_sha256=hashlib.sha256(np.ascontiguousarray(batch['ids']).tobytes()).hexdigest()))

            if arm == 'uniform':
                if not resume:
                    loader_seed_draw()
                interval = max(1, config['diagnosis']['window_steps'])
                while clock['actual_step'] < endpoint:
                    update(next(sampler), 'normal', False)
                    if clock['actual_step'] % interval == 0 or clock['actual_step'] == endpoint:
                        clock['window'] = clock['actual_step'] // interval
                        checkpoint()
                        write_json(out / 'status.json', clock)
            else:
                normal_steps = config['diagnosis']['window_steps']
                while clock['actual_step'] < endpoint:
                    if progress['phase'] == 'start':
                        progress.update(phase='probe', order=np.concatenate([
                            np.random.permutation(len(corpus.domains))
                            for _ in range(pr['rounds'])]).tolist(), target_index=0,
                            probe_batch_index=0, normal_done=0, segment_sampler_state=None)
                    if progress['phase'] == 'probe':
                        while progress['target_index'] < len(progress['order']):
                            target = progress['order'][progress['target_index']]
                            if progress['segment_sampler_state'] is None:
                                progress['segment_sampler_state'] = sampler.state_dict()
                                segment = sampler.segment(target, pr['one_hot_factor'],
                                                          pr['steps'], tr['batch_size'])
                                loader_seed_draw()
                            else:
                                # Rebuild a partially consumed segment without advancing its RNG twice.
                                current_sampler_state = sampler.state_dict()
                                sampler.load_state_dict(progress['segment_sampler_state'])
                                try:
                                    segment = sampler.segment(target, pr['one_hot_factor'],
                                                              pr['steps'], tr['batch_size'])
                                finally:
                                    sampler.load_state_dict(current_sampler_state)
                            observed = pr['use_history'] and segment.metadata_supported
                            for batch in segment.batches[progress['probe_batch_index']:]:
                                if clock['actual_step'] == endpoint:
                                    break
                                if observed:
                                    refresh_targets()
                                update(batch, 'probe', observed)
                                diag.note_excluded_probe_updates(1, draws=len(batch['ids']))
                                progress['probe_batch_index'] += 1
                            if progress['probe_batch_index'] < segment.actual_updates:
                                break
                            append_json(logs['probes'], {'window': clock['window'],
                                                        **segment.audit_record()})
                            progress['target_index'] += 1
                            progress['probe_batch_index'] = 0
                            progress['segment_sampler_state'] = None
                            if clock['actual_step'] == endpoint:
                                break
                        if progress['target_index'] == len(progress['order']):
                            sampler.mirror_normal_rng()
                            loader_seed_draw()
                            progress['phase'] = 'normal'
                    if progress['phase'] == 'normal':
                        while progress['normal_done'] < normal_steps and clock['actual_step'] < endpoint:
                            refresh_targets()
                            batch = corpus.draw(controller.cut, controller.weights, tr['batch_size'])
                            update(batch, 'normal', True)
                            progress['normal_done'] += 1
                        if progress['normal_done'] == normal_steps:
                            boundary = diag.complete_window()
                            decision = controller.update(boundary.forecast)
                            append_json(logs['boundaries'], {'nominal_step': clock['actual_step'],
                                'actual_step': clock['actual_step'], 'audit': boundary.audit,
                                'forecast': boundary.forecast, 'decision': decision})
                            clock['window'] += 1
                            progress.update(phase='start', order=[], target_index=0,
                                            probe_batch_index=0, normal_done=0,
                                            segment_sampler_state=None)
                            checkpoint()
                            write_json(out / 'status.json', clock)
                            print(json.dumps(plain(clock)), flush=True)
                checkpoint()
                write_json(out / 'status.json', clock)
        if (scheduler.last_epoch != clock['actual_step'] or clock['actual_step'] != endpoint
                or clock['nominal_step'] != clock['actual_step']
                or clock['probe_updates'] + clock['normal_updates'] != clock['actual_step']):
            raise RuntimeError('Optimizer, scheduler, and update budgets disagree')
        evaluations = {}
        if not skip_evaluation:
            for split in (('dev', 'test') if evaluate_test else ('dev',)):
                evaluations[split] = evaluate(model, corpus, split=split, pad_id=tr['pad_token_id'],
                    batch_size=tr['eval_batch_size'], validation_fraction=config['target']['validation_fraction'])
        result = {**clock, 'complete': clock['actual_step'] == tr['max_steps'], 'arm': arm,
                  'synthetic_fixture': model_factory is not None, 'evaluations': evaluations,
                  'elapsed_seconds': time.monotonic() - started,
                  'checkpoint': str(out / 'checkpoints' / f"step_{clock['actual_step']:06d}.pt")}
        write_json(out / 'summary.json', result)
        write_json(out / 'status.json', result)
        return result
    except BaseException as exc:
        write_json(out / 'failure.json', {**clock, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        if ghost is not None:
            ghost.close()
