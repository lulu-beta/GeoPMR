import time
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk

from .hyper_hier_module import HHM, PoincareBall


def _sigmoid_bounded_param(raw: torch.Tensor, max_val: float) -> torch.Tensor:
    """Map an unconstrained scalar to (0, max_val) with a smooth bound."""
    return max_val * torch.sigmoid(raw)


def context_propagation_recon_loss(pred, target, weight, nan_mask, mode="auto"):
    """
    Context Propagation Reconstruction (CPR) Loss.

    This function adaptively selects the reconstruction loss based on the modality type:
        - Molecular modality (binary data): Binary Cross Entropy with logits
        - Continuous modalities (gene expression, morphology, etc.): Smooth L1 (Huber) loss or MSE
    """
    expanded_weight = weight.unsqueeze(2).expand_as(target)
    pred_valid = pred[~nan_mask]
    target_valid = target[~nan_mask]
    w = expanded_weight[~nan_mask]

    if mode == "auto":
        is_binary = (
            target_valid.min().item() >= 0.0 and
            target_valid.max().item() <= 1.0 and
            target_valid.unique().numel() <= 10
        )
    else:
        is_binary = (mode == "binary")

    if is_binary:
        loss = F.binary_cross_entropy_with_logits(pred_valid, target_valid, weight=w, reduction="none")
    else:
        loss = F.smooth_l1_loss(torch.sigmoid(pred_valid), target_valid, reduction="none")
        loss = loss * w

    return loss.mean()




class DiagonalMetricNet(nn.Module):
    """
    Lightweight learnable diagonal PSD metric for MGW-style alignment.

    Given an embedding z (B, D), outputs per-dimension positive weights w (B, D),
    defining a diagonal metric g = diag(w).
    """

    def __init__(self, dim: int, hidden: int = 256, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z)) + self.eps


def _pairwise_sqdist_diagonal_metric(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances under a symmetrised per-point diagonal metric.

    Uses the average of the two endpoint metrics: 0.5*(w_i + w_j), so the
    distance d(i,j) == d(j,i).
    """
    diff = z.unsqueeze(1) - z.unsqueeze(0)  # (N, N, D)
    w_sym = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0))  # (N, N, D)
    return (diff * diff * w_sym).sum(dim=-1)


def mgw_entropic_gw_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    metric1: "DiagonalMetricNet",
    metric2: "DiagonalMetricNet",
    *,
    subsample: int = 128,
    epsilon: float = 0.1,
    outer_iters: int = 5,
    sinkhorn_iters: int = 30,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MGW-inspired alignment regularizer:
      learn modality-specific metrics -> compute induced distances -> entropic GW.

    This adapts the MGW objective to GeoPMR latent spaces.
    """
    device = z1.device
    n_full = z1.size(0)
    if subsample is not None and subsample > 0 and n_full > subsample:
        idx = torch.randperm(n_full, device=device)[:subsample]
        x = z1[idx]
        y = z2[idx]
    else:
        x, y = z1, z2

    n = x.size(0)
    if n < 3:
        return torch.tensor(0.0, device=device)

    wx = metric1(x)  # (n, D)
    wy = metric2(y)  # (n, D)

    C1 = _pairwise_sqdist_diagonal_metric(x, wx)
    C2 = _pairwise_sqdist_diagonal_metric(y, wy)
    C1 = C1 / (C1.max() + eps)
    C2 = C2 / (C2.max() + eps)

    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((n,), 1.0 / n, device=device)

    T = torch.outer(a, b)  # (n, n)
    C1_sq = C1 * C1
    C2_sq = C2 * C2

    for _ in range(outer_iters):
        const1 = C1_sq @ a  # (n,)
        const2 = C2_sq @ b  # (n,)
        M = const1.unsqueeze(1) + const2.unsqueeze(0) - 2.0 * (C1 @ T @ C2.t())  # (n, n)

        K = torch.exp(-M / epsilon)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(sinkhorn_iters):
            u = a / (K @ v + eps)
            v = b / (K.t() @ u + eps)
        T = torch.diag(u) @ K @ torch.diag(v)
        T = T / (T.sum() + eps)

    return (M * T).sum()


def semantic_consistency_alignment_loss(
    zi: torch.Tensor,
    zj: torch.Tensor,
    temperature: float = 0.2,
    mode: str = "info_nce",  # "info_nce" or "vicreg"
    invar: float = 1.0,
    var: float = 1.0,
    cov: float = 1.0,
    eps: float = 1e-4,
):
    """
    Semantic Consistency Alignment Loss.

    Computes alignment loss between two projected modalities.
    Supports InfoNCE (cross-modal matching) and VICReg (augmentation consistency) styles.

    Args:
        zi, zj (Tensor): (B, D) projected features from two modalities
        temperature (float): temperature scaling for InfoNCE
        mode (str): "info_nce" or "vicreg"
        invar, var, cov (float): VICReg weights
        eps (float): numerical stability for std
    Returns:
        loss (Tensor): computed alignment loss
    """
    if mode == "info_nce":
        # === InfoNCE style ===
        z_a = F.normalize(zi, dim=-1)
        z_b = F.normalize(zj, dim=-1)
        logits = z_a @ z_b.t() / temperature  # (B, B)
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, labels)

    elif mode == "vicreg":
        # === VICReg style ===
        # 1. Invariance loss (MSE)
        invariance_loss = F.mse_loss(zi, zj)

        # 2. Variance regularization
        def variance_loss(x):
            std = torch.sqrt(x.var(dim=0) + eps)
            return torch.mean(F.relu(1.0 - std))

        var_loss = variance_loss(zi) + variance_loss(zj)

        # 3. Covariance regularization
        def covariance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / (x.size(0) - 1)
            cov_no_diag = cov - torch.diag_embed(torch.diagonal(cov))
            return (cov_no_diag ** 2).sum() / x.size(1)

        cov_loss = covariance_loss(zi) + covariance_loss(zj)

        return invar * invariance_loss + var * var_loss + cov * cov_loss

    else:
        raise ValueError(f"Unknown alignment mode: {mode}")




def inter_modal_hierarchy_loss(hhm_loss, weight=1.0):
    """
    inter-modal consistency
    """
    return weight * hhm_loss



def pretrain_func(args, model, loaders, aligned_data, aug_data, ctx_graph, optimizer, scheduler, epoch):
    """
    Pre-training function for multi-modal representation learning.
    Includes:
        - Context-Propagation Reconstruction (CPR) loss
        - Intra-Modal Alignment loss
        - Inter-Modal Hierarchy loss
    """
    model.train()
    device = args.device

    from utils.misc import AverageMeter
    meters = {k: AverageMeter() for k in ["total", "cpr", "intra", "inter"]}

    mol_id_to_idx = {k: i for i, k in enumerate(aligned_data["mol_id"])}

    # === Constants from context graph ===
    edge_w = torch.cat([ctx_graph.weight,
                        torch.zeros(1, dtype=ctx_graph.weight.dtype, device=ctx_graph.weight.device)])
    idx_map = {t: i for i, t in enumerate(ctx_graph.type)}

    # Progress bar
    pbar = tqdm(range(args.steps), disable=args.no_print)


    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(loaders["train_iter"])
        except StopIteration:
            loaders["train_iter"] = iter(loaders["train_loader"])
            batch = next(loaders["train_iter"])
        batch = batch.to(device)

        if batch.x.size(0) == 1 or batch.batch[-1] == 0:
            continue

        

        # === Gather modality inputs ===
        input1D = batch.feature_1D.float()
        input2D = model.pool(model.graph_encoder(batch)[0], batch.batch)
        input3D = batch.feature_3D.float()

        mol_ids = batch.type
        selected_idx = [mol_id_to_idx[mid] for mid in mol_ids]

        gene_crispr_emb = torch.tensor(aligned_data["crispr_feat"][selected_idx], device=device, dtype=torch.float32)
        gene_orf_emb = torch.tensor(aligned_data["orf_feat"][selected_idx], device=device, dtype=torch.float32)
        input_gene = torch.cat([gene_crispr_emb, gene_orf_emb], dim=1)

        cell_cp_bray_emb = torch.tensor(aligned_data["cp_bray_feat"][selected_idx], device=device, dtype=torch.float32)
        cell_cp_jump_emb = torch.tensor(aligned_data["cp_jump_feat"][selected_idx], device=device, dtype=torch.float32)
        input_cell = torch.cat([cell_cp_bray_emb, cell_cp_jump_emb], dim=1)

        input_express = torch.tensor(aligned_data["express_feat"][selected_idx], device=device, dtype=torch.float32)

        modal_inputs = [input1D, input2D, input3D, input_gene, input_cell, input_express]


        gene_crispr_emb_aug = torch.tensor(aug_data["crispr_feat"][selected_idx], device=device, dtype=torch.float32)
        gene_orf_emb_aug = torch.tensor(aug_data["orf_feat"][selected_idx], device=device, dtype=torch.float32)
        input_gene_aug  = torch.cat([gene_crispr_emb_aug, gene_orf_emb_aug], dim=1)

        cell_cp_bray_emb_aug = torch.tensor(aug_data["cp_bray_feat"][selected_idx], device=device, dtype=torch.float32)
        cell_cp_jump_emb_aug = torch.tensor(aug_data["cp_jump_feat"][selected_idx], device=device, dtype=torch.float32)
        input_cell_aug = torch.cat([cell_cp_bray_emb_aug, cell_cp_jump_emb_aug], dim=1)

        input_express_aug = torch.tensor(aug_data["express_feat"][selected_idx], device=device, dtype=torch.float32)

        modal_inputs_aug = [input_gene_aug, input_cell_aug, input_express_aug]


        # === Graph random walk (for CPR supervision) ===
        start_idx = torch.tensor([idx_map[t] for t in batch.type])
        walk, edge_seq = random_walk(ctx_graph.edge_index[0], ctx_graph.edge_index[1],
                                     start_idx, args.walk_length,
                                     num_nodes=ctx_graph.num_nodes, return_edge_indices=True)
        path_w = edge_w[edge_seq].view(-1, args.walk_length).cumprod(-1).to(device)
        path_w = torch.cat([torch.ones_like(path_w[:, :1]), path_w], dim=1)  # include start step

        # === Targets from graph propagation ===

        tgt_mol = ctx_graph.mol_target[walk].to(device)
        tgt_gene = ctx_graph.gene_target[walk].to(device)
        tgt_cell = ctx_graph.cell_target[walk].to(device)
        tgt_expr = ctx_graph.express_target[walk].to(device)
        masks = {"mol": torch.isnan(tgt_mol),
                 "gene": torch.isnan(tgt_gene),
                 "cell": torch.isnan(tgt_cell),
                 "expr": torch.isnan(tgt_expr)}



        # === Forward pass ===
        latent_codes, latent_vectors, hhm_loss, modal_latents, anchor_latent, projected_aug, modal_reconstructions = model(modal_inputs, modal_inputs_aug)




        # === Compute losses ===
        # --- (1) Context-Propagation Reconstruction Loss (CPR) ---
        expand = lambda p, tgt: p.unsqueeze(1).expand(-1, tgt.size(1), -1)
        preds = [expand(p, t) for p, t in zip(modal_reconstructions, [tgt_mol, tgt_gene, tgt_cell, tgt_expr])]
        targets = [tgt_mol, tgt_gene, tgt_cell, tgt_expr]
        modality_keys = ["mol", "gene", "cell", "expr"]

        cpr_loss = 0.0
        for pred, tgt, key in zip(preds, targets, modality_keys):
            cpr_loss += context_propagation_recon_loss(pred, tgt, path_w, masks[key])

        # --- (2) Intra-Modal Alignment Loss ---
        intra_loss = 0.0
        temperature = getattr(args, "temperature", 0.1)

        # --- MGW alpha values (computed once, outside the loop) ---
        if getattr(args, "mgw_learn_alpha", False):
            mgw_alpha     = model.mgw_alpha()
            mgw_alpha_aug = model.mgw_alpha_aug()
        else:
            mgw_alpha     = float(getattr(args, "mgw_alpha", 0.0))
            mgw_alpha_aug = float(getattr(args, "mgw_alpha_aug", 0.0))
        do_mgw     = (mgw_alpha.item()     if torch.is_tensor(mgw_alpha)     else mgw_alpha)     > 1e-12
        do_mgw_aug = (mgw_alpha_aug.item() if torch.is_tensor(mgw_alpha_aug) else mgw_alpha_aug) > 1e-12

        # --- (2a) Anchor ↔ Biological InfoNCE ---
        w_infonce = model.infonce_weight() if getattr(args, "learn_alignment_weights", False) else 1.0
        for k, j in enumerate(range(3, len(modal_latents))):
            infonce_loss = semantic_consistency_alignment_loss(
                anchor_latent, modal_latents[j], temperature=temperature, mode="info_nce",
            )
            intra_loss += w_infonce * infonce_loss

            if do_mgw:
                ot_reg = mgw_entropic_gw_loss(
                    anchor_latent, modal_latents[j],
                    model.mgw_metric_anchor, model.mgw_metric_bio[k],
                    subsample=getattr(args, "mgw_subsample", 128),
                    epsilon=getattr(args, "mgw_epsilon", 0.1),
                    outer_iters=getattr(args, "mgw_outer_iters", 3),
                    sinkhorn_iters=getattr(args, "mgw_sinkhorn_iters", 20),
                )
                intra_loss += mgw_alpha * ot_reg

        # --- (2b) Augmentation VICReg consistency ---
        w_vicreg = model.vicreg_weight() if getattr(args, "learn_alignment_weights", False) else 1.0
        for k, j in enumerate(range(3, len(modal_latents))):
            vicreg_loss = semantic_consistency_alignment_loss(
                modal_latents[j], projected_aug[k], mode="vicreg",
            )
            intra_loss += w_vicreg * vicreg_loss

            if do_mgw_aug:
                ot_reg_aug = mgw_entropic_gw_loss(
                    modal_latents[j], projected_aug[k],
                    model.mgw_metric_bio[k], model.mgw_metric_bio[k],
                    subsample=getattr(args, "mgw_subsample", 128),
                    epsilon=getattr(args, "mgw_epsilon", 0.1),
                    outer_iters=getattr(args, "mgw_outer_iters", 3),
                    sinkhorn_iters=getattr(args, "mgw_sinkhorn_iters", 20),
                )
                intra_loss += mgw_alpha_aug * ot_reg_aug

        # --- (3) Inter-Modal Hierarchical Loss ---
        inter_loss = inter_modal_hierarchy_loss(hhm_loss)

        # --- Total Loss ---
        total_loss = (
            cpr_loss +
            args.lambda_1 * intra_loss +
            args.lambda_2 * inter_loss
        )

        # === Backward and optimize ===
        total_loss.backward()
        grad_clip = getattr(args, 'grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        # === Update meters ===
        meters["total"].update(total_loss.item())
        meters["cpr"].update(cpr_loss.item())
        meters["intra"].update(intra_loss.item())
        meters["inter"].update(inter_loss.item())


        pbar.set_description(
            f"Ep {epoch+1}/{args.epochs} | It {step+1}/{args.steps} | "
            f"Tot {meters['total'].avg:.3f} "
            f"(CPR {meters['cpr'].avg:.3f}, Intra {meters['intra'].avg:.3f}, Inter {meters['inter'].avg:.3f})"
        )

    pbar.close()
    if not args.no_print:
        args.logger.info(
            f"Epoch {epoch+1}: "
            f"Total {meters['total'].avg:.4f}, "
            f"CPR {meters['cpr'].avg:.4f}, "
            f"Intra {meters['intra'].avg:.4f}, "
            f"Inter {meters['inter'].avg:.4f}"
        )
    return meters["total"].avg, loaders


class GNN(torch.nn.Module):
    def __init__(self, *, num_layer=5, emb_dim=300, gnn_type="gin", drop_ratio=0.5,
                 graph_pooling="max", norm_layer="batch_norm",
                 pro_dims=None, decoder_dims=None, depth=6, ec_ce_weight=1.0,
                 hierarchy_temperature=0.1,
                 hierarchy_lambda_sim=1.0, hierarchy_lambda_dissim=0.5, hyperbolic_curvature=1.0):
        super().__init__()
        self.hierarchy_lambda_sim = hierarchy_lambda_sim
        self.hierarchy_lambda_dissim = hierarchy_lambda_dissim
        self.hyperbolic_curvature = hyperbolic_curvature
        decoder_dims = decoder_dims or [1024, 1111+862, 1783+966, 978]
        pro_dims = pro_dims or [1024+167, 300, 512, 1111+862, 1783+966, 978]  # mol, gene, cell, expr dims  [300, 1111, 862, 1783, 966, 978, 1024+167, 512]

        from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
        from .conv import GNN_node, GNN_node_Virtualnode

        gnn_name = gnn_type.split("-")[0]
        node_cls = GNN_node_Virtualnode if "virtual" in gnn_type else GNN_node

        self.graph_encoder = node_cls(num_layer, emb_dim, JK="last", drop_ratio=drop_ratio,
                                      residual=True, gnn_name=gnn_name, norm_layer=norm_layer)

        pool_map = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
        if graph_pooling not in pool_map:
            raise ValueError("Invalid graph pooling type")
        self.pool = pool_map[graph_pooling]

        self.modal_projectors = torch.nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, emb_dim * 2),
                nn.SiLU(),
                nn.LayerNorm(emb_dim * 2),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.SiLU(),
            ) for dim in pro_dims
        ])


        self.modal_decoders = nn.ModuleList([
            MLP(emb_dim, hidden_features=emb_dim * 4, out_features=d) for d in decoder_dims
        ])

        # === MGW-style metric learners (optional regularizer) ===
        # Anchor metric (for anchor_latent) + one metric per biological modality (gene/cell/expr).
        # These are used by mgw_entropic_gw_loss when args.mgw_alpha > 0.
        self.mgw_metric_anchor = DiagonalMetricNet(dim=emb_dim, hidden=256)
        self.mgw_metric_bio = nn.ModuleList([DiagonalMetricNet(dim=emb_dim, hidden=256) for _ in range(3)])

        # Learnable weights for MGW regularizers (optional; enabled via args.mgw_learn_alpha)
        # We store raw (unconstrained) parameters and map them to (0, mgw_alpha_max) with sigmoid.
        self.mgw_alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.mgw_alpha_aug_raw = nn.Parameter(torch.tensor(0.0))
        self.mgw_alpha_max = 0.1  # default; overwritten from args in pretrain.py via setattr if desired

        # Learnable weights for InfoNCE (cross-modal) and VICReg (augmentation) (optional; --learn-alignment-weights)
        self.infonce_weight_raw = nn.Parameter(torch.tensor(0.0))
        self.vicreg_weight_raw = nn.Parameter(torch.tensor(0.0))
        self.alignment_weight_max = 2.0  # default; overwritten from args in pretrain.py if desired

        # Learnable per-view weights for computing the molecular anchor embedding
        self.anchor_weights = nn.Parameter(torch.zeros(3))

        # Shared PoincareBall instance
        self.poincare_ball = PoincareBall(dim=emb_dim, c=hyperbolic_curvature)

        # Hierarchy: 1D (0) → 2D (1) → 3D (2) → Gene (3) → Cell (4) → Expression (5)
        hierarchy_structure = [0, 1, 2, 3, 4, 5]
        self.hyper_hierarchical = HHM(
            num_modalities=len(pro_dims),
            latent_dim=emb_dim,
            hierarchy_structure=hierarchy_structure,
            temperature=hierarchy_temperature,
            lambda_sim=hierarchy_lambda_sim,
            lambda_dissim=hierarchy_lambda_dissim,
            c=hyperbolic_curvature,
        )

    def forward(self, modal_inputs, modal_inputs_aug=None):
        """
        Args:
            modal_inputs: List of tensors for each modality.
        Returns:
            - latent_codes: Hierarchical latent indices per modality
            - modal_latents: Projected modal features
            - modal_reconstructions: Decoded modal outputs
        """
        # Project modalities into shared embedding space
        modal_latents = [proj(x) for proj, x in zip(self.modal_projectors, modal_inputs)]

        projected_aug = None
        if modal_inputs_aug is not None:
            projected_aug = [
                proj(x_aug) for proj, x_aug in zip(self.modal_projectors[3:], modal_inputs_aug)
            ]

        # Compute weighted anchor from the three molecular views
        w = torch.softmax(self.anchor_weights, dim=0)
        anchor_latent = sum(w[i] * modal_latents[i] for i in range(3))

        # Decode modalities
        modal_reconstructions = [
            decoder(x) for decoder, x in zip(
                self.modal_decoders,
                [anchor_latent, modal_latents[3], modal_latents[4], modal_latents[5]]
            )
        ]

        # HHM loss
        latent_codes, latent_vectors, hhm_loss = self.hyper_hierarchical(modal_latents)

        return latent_codes, latent_vectors, hhm_loss, modal_latents, anchor_latent, projected_aug, modal_reconstructions

    def mgw_alpha(self) -> torch.Tensor:
        return _sigmoid_bounded_param(self.mgw_alpha_raw, float(self.mgw_alpha_max))

    def mgw_alpha_aug(self) -> torch.Tensor:
        return _sigmoid_bounded_param(self.mgw_alpha_aug_raw, float(self.mgw_alpha_max))

    def infonce_weight(self) -> torch.Tensor:
        return _sigmoid_bounded_param(self.infonce_weight_raw, float(self.alignment_weight_max))

    def vicreg_weight(self) -> torch.Tensor:
        return _sigmoid_bounded_param(self.vicreg_weight_raw, float(self.alignment_weight_max))

    # ---- utility: load & freeze graph encoder ----
    def load_pretrained_graph_encoder(self, path: str):
        state = torch.load(path, map_location="cpu")
        ge_state = {k.replace("graph_encoder.", ""): v for k, v in state.items() if k.startswith("graph_encoder.")}
        self.graph_encoder.load_state_dict(ge_state)

    def freeze_graph_encoder(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad_(False)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,  # >>> MOD: GELU over SiLU.
        bias: bool = True,
        drop: float = 0.3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features, bias=bias),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net(x)
    
