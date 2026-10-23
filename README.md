# GranMix

Code for GranMix


## Environment

Use Python 3.10. For GPU training, use the CUDA 12.1 build of PyTorch 2.2.1.

```bash
cd code
pip install -r requirements.txt
```

## Data

Pass the tokenized corpus root with `--data`. Each domain contains
`train.bin`, `val.bin`, and `test.bin` as uint16 token streams.
Tokenization and row order must match `assets/identity.json` and the bundled
tree labels. Large corpus files and trained checkpoints are not included.
Cases: `ase`, `ghc4`, `bse`, `abse`, `ccghw`, and `seven`.

## Usage

Run from `code/`:

```bash
python main.py inspect --case ase
python main.py preflight --case ase --data /path/to/corpus --full-hash

CUDA_VISIBLE_DEVICES=0 python main.py train \
  --case ase --data /path/to/corpus \
  --output /path/to/new_run --evaluate-test

python main.py train --case ase --data /path/to/corpus \
  --resume /path/to/new_run/checkpoints/step_000250.pt \
  --output /path/to/resumed_run --evaluate-test

python main.py evaluate \
  --checkpoint /path/to/resumed_run/checkpoints/step_005000.pt \
  --data /path/to/corpus --split test --output /path/to/test.json
```

GranMix is the default; use `--arm uniform` for the matched control.
Outputs must use new paths. For a partial run, add `--end 250 --skip-evaluation`.
Resume at a complete GranMix window boundary using a checkpoint from this code.
