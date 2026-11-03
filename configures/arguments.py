#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument definitions for GeoPMR training and evaluation."""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="GeoPMR pre-training / fine-tuning")

    # Runtime / IO
    parser.add_argument("--gpu-id", type=int, default=0, help="Which GPU to use (default: 0)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no-print", action="store_true", default=False, help="Disable progress bar/log prints")
    parser.add_argument("--dataset", type=str, default="pretrain", help="Dataset name")
    parser.add_argument("--model-path", type=str, default="pretrain.pt", help="Path to save/load model checkpoint")

    # Model / GNN
    parser.add_argument("--model", type=str, default="gin-virtual",
                        help="Model type: gin, gin-virtual, gcn, gcn-virtual")
    parser.add_argument("--readout", type=str, default="sum", help="Graph readout function")
    parser.add_argument("--norm-layer", type=str, default="batch_norm", help="Normalization layer type")
    parser.add_argument("--drop-ratio", type=float, default=0.5, help="Dropout ratio")
    parser.add_argument("--num-layer", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--emb-dim", type=int, default=300, help="Hidden dimension in GNNs")

    # Context graph
    parser.add_argument("--walk-length", type=int, default=4, help="Context walk length")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for context graph")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=5120, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--initw-name", type=str, default="default", help="Weight initialization method")

    # Preprocessing
    parser.add_argument("--fill_method", type=str, default="mean", help="Fill method: mean, zero, nearest")

    # Alignment temperature (InfoNCE)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Initial / fixed temperature for InfoNCE. When --learn-temperature is on, "
                             "this is the initialisation value.")
    parser.add_argument("--learn-temperature", action="store_true", default=False,
                        help="If set, the InfoNCE temperature is a learnable parameter (CLIP-style).")
    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max norm for gradient clipping (0 disables).")
    # Warmup fraction
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps used for cosine warmup (default: 0.1 = 10%%).")

    # Task-specific (keep original defaults)
    parser.add_argument("--depth", type=int, default=6)                 # d
    parser.add_argument("--lambda_1", type=float, default=0.1)
    parser.add_argument("--lambda_2", type=float, default=0.01)
    parser.add_argument("--ec_ce_weight", type=float, default=1.0)      # μ
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--task_dropout", type=float, default=0.8)
    
    # Hyperbolic hierarchy
    parser.add_argument("--use-hyperbolic-hierarchy", action="store_true", default=True,
                        help="Hyperbolic hierarchy is always enabled; retained for compatibility.")
    parser.add_argument("--hierarchy-temperature", type=float, default=0.1,
                        help="Temperature for hierarchical contrastive loss")
    parser.add_argument("--hierarchy-lambda-sim", type=float, default=1.0,
                        help="Weight for parent-child similarity loss")
    parser.add_argument("--hierarchy-lambda-dissim", type=float, default=0.5,
                        help="Weight for sibling dissimilarity loss")
    parser.add_argument("--hierarchy-lambda-kl", type=float, default=0.01,
                        help="Weight for KL divergence regularization loss")
    parser.add_argument("--hyperbolic-curvature", type=float, default=1.0,
                        help="Curvature of hyperbolic space")
    
    # MGW-style alignment regularizer (metric learning + entropic GW on induced distances)
    parser.add_argument("--mgw-alpha", type=float, default=0.0,
                        help="Weight for MGW-style GW alignment regularizer (0 disables). Try 0.001-0.05")
    parser.add_argument("--mgw-alpha-aug", type=float, default=0.0,
                        help="Weight for MGW-style GW regularizer on augmentation pairs (0 disables). Try 0.001-0.05")
    parser.add_argument("--mgw-learn-alpha", action="store_true", default=False,
                        help="If set, learn mgw-alpha and mgw-alpha-aug as trainable scalars (initialized from the provided values).")
    parser.add_argument("--mgw-alpha-max", type=float, default=0.1,
                        help="Max value for learned MGW alphas when --mgw-learn-alpha is enabled (bounded with sigmoid).")
    parser.add_argument("--mgw-subsample", type=int, default=128,
                        help="Subsample size for MGW GW computation (controls O(n^2) memory/time)")
    parser.add_argument("--mgw-epsilon", type=float, default=0.1,
                        help="Entropic regularization epsilon for GW Sinkhorn")
    parser.add_argument("--mgw-outer-iters", type=int, default=3,
                        help="Number of outer GW iterations")
    parser.add_argument("--mgw-sinkhorn-iters", type=int, default=20,
                        help="Number of Sinkhorn iterations per GW outer step")

    # Learnable alignment weights (InfoNCE cross-modal and VICReg augmentation)
    parser.add_argument("--learn-alignment-weights", action="store_true", default=False,
                        help="If set, learn weights for InfoNCE cross-modal and VICReg augmentation as trainable scalars.")
    parser.add_argument("--infonce-weight", type=float, default=1.0,
                        help="Initial weight for InfoNCE (used when --learn-alignment-weights).")
    parser.add_argument("--vicreg-weight", type=float, default=1.0,
                        help="Initial weight for VICReg augmentation (used when --learn-alignment-weights).")
    parser.add_argument("--alignment-weight-max", type=float, default=2.0,
                        help="Max value for learned alignment weights (bounded with sigmoid).")

    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to append to model name."
    )
    parser.add_argument("--timestamp", type=str, default="", help="Timestamp to locate unique checkpoint.")

    args = parser.parse_args()

    # Extra default parameter preserved from your code
    args.n_steps = 1

    return args

