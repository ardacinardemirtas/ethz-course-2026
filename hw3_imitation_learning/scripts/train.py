"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3 # 3e-3
VAL_SPLIT = 0.1


def split_by_episodes(
    states: np.ndarray,
    actions: np.ndarray,
    ep_ends: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split states/actions into train and val by episode to avoid data leakage.

    Chunks from the same episode are kept together so no episode appears in
    both splits.
    """
    n_ep = len(ep_ends)
    ep_starts = np.concatenate(([0], ep_ends[:-1]))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(n_ep * val_fraction))

    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    def gather(indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_parts: list[np.ndarray] = []
        a_parts: list[np.ndarray] = []
        ends: list[int] = []
        offset = 0
        for i in indices:
            s, e = int(ep_starts[i]), int(ep_ends[i])
            s_parts.append(states[s:e])
            a_parts.append(actions[s:e])
            offset += e - s
            ends.append(offset)
        if not s_parts:
            return states[:0], actions[:0], np.array([], dtype=np.int64)
        return (
            np.concatenate(s_parts),
            np.concatenate(a_parts),
            np.array(ends, dtype=np.int64),
        )

    return gather(train_idx), gather(val_idx)


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--d-model", type=int, default=512, help="Hidden layer width (default: 512).")
    parser.add_argument("--depth", type=int, default=4, help="Number of hidden layers (default: 4).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (default: 0.1).")
    parser.add_argument("--full-state", action="store_true", help="Pass full state to MultiTaskPolicy instead of the reduced 6-dim target-centric input.")
    parser.add_argument("--relative-state", action="store_true", help="Use relative state representation for MultiTaskPolicy: (ee-bin)(3), (ee-cube)(3), gripper(1).")
    parser.add_argument(
        "--extra-zarr",
        nargs="+",
        type=Path,
        default=None,
        help="Additional .zarr stores to merge with --zarr.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )

    normalizer = Normalizer.from_data(states, actions)
    dataset = SO100ChunkDataset(states, actions, ep_ends, chunk_size=args.chunk_size, normalizer=normalizer)

    # ── train / val split ─────────────────────────────────────────────
    # Note: sample-level split intentionally used here. Episode-aware splitting
    # requires ~50+ episodes to produce a reliable val set; with small datasets
    # the noise from 1-2 val episodes makes early stopping unreliable.
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Dataset: {len(dataset)} samples ({len(ep_ends)} episodes) → {len(train_ds)} train / {len(val_ds)} val, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        use_full_state=args.full_state,
        use_relative_state=args.relative_state,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "dropout": args.dropout,
                    "use_layer_norm": True,
                    "use_full_state": args.full_state,
                    "use_relative_state": args.relative_state,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
