import os, json, argparse, hashlib
import pandas as pd

# Deterministic 60/20/20 split stored for reuse.
# Output: data/splits.json with keys train, val, test listing subject_ids.

def parse_args():
    p = argparse.ArgumentParser(description='Prepare persistent 60/20/20 train/val/test splits.')
    p.add_argument('--labels', required=True, help='Path to labels.csv with subject_id and readmission_label (plus others).')
    p.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'data', 'splits.json'), help='Output JSON path.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.labels)
    if 'subject_id' not in df.columns:
        raise SystemExit('labels file must include subject_id')
    # To ensure deterministic ordering before split
    df = df.sort_values('subject_id').reset_index(drop=True)
    rng = pd.Series(df['subject_id']).sample(frac=1.0, random_state=args.seed).tolist()
    n = len(rng)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_ids = rng[:n_train]
    val_ids = rng[n_train:n_train+n_val]
    test_ids = rng[n_train+n_val:]
    splits = {
        'seed': args.seed,
        'counts': {'train': len(train_ids), 'val': len(val_ids), 'test': len(test_ids), 'total': n},
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'hash': hashlib.sha256((','.join(map(str, rng))).encode()).hexdigest()[:16],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print('Wrote splits to', args.output)
    print('Counts:', splits['counts'])

if __name__ == '__main__':
    main()
