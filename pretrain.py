import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from configures.arguments import get_args
from dataset.create_datasets import get_data
from dataset.data_utils import align_and_fill_modalities, align_and_aug_modalities
from utils import init_weights
from models.pretrain_model import pretrain_func
from utils.training_utils import get_logger, get_cosine_schedule_with_warmup


from models.pretrain_model import GNN
from torch.distributions import Normal, Independent

def _logit(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    dataset, context_graph = get_data(args, "./raw_data", transform="pyg")
    aligned_data = align_and_fill_modalities("./raw_data/pretrain/raw", fill_method=args.fill_method)
    aug_data = align_and_aug_modalities("./raw_data/pretrain/raw")
    context_graph = context_graph[0]


    split_idx = dataset.get_idx_split()
    args.num_trained = len(split_idx["train"])
    args.task_type = dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = GNN(
        gnn_type=args.model,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.readout,
        norm_layer=args.norm_layer,
        depth=args.depth,
        ec_ce_weight=args.ec_ce_weight,
        hierarchy_temperature=getattr(args, 'hierarchy_temperature', 0.1),
        hierarchy_lambda_sim=getattr(args, 'hierarchy_lambda_sim', 1.0),
        hierarchy_lambda_dissim=getattr(args, 'hierarchy_lambda_dissim', 0.5),
        hyperbolic_curvature=getattr(args, 'hyperbolic_curvature', 1.0),
    ).to(device)
    # init_weights BEFORE loading pretrained encoder so the encoder weights are not overwritten
    init_weights(model, args.initw_name, init_gain=0.02)
    model.load_pretrained_graph_encoder("./ckpt/pretrain2D.pt")

    # Optionally learn MGW weights (mgw-alpha, mgw-alpha-aug).
    if getattr(args, "mgw_learn_alpha", False):
        alpha_max = float(getattr(args, "mgw_alpha_max", 0.1))
        model.mgw_alpha_max = alpha_max

        init_alpha = float(getattr(args, "mgw_alpha", 0.0))
        init_alpha_aug = float(getattr(args, "mgw_alpha_aug", 0.0))

        frac = torch.tensor(init_alpha / max(alpha_max, 1e-12), device=device)
        frac_aug = torch.tensor(init_alpha_aug / max(alpha_max, 1e-12), device=device)

        model.mgw_alpha_raw.data = _logit(frac).detach()
        model.mgw_alpha_aug_raw.data = _logit(frac_aug).detach()

    # Optionally learn InfoNCE and VICReg alignment weights.
    if getattr(args, "learn_alignment_weights", False):
        align_max = float(getattr(args, "alignment_weight_max", 2.0))
        model.alignment_weight_max = align_max

        init_infonce = float(getattr(args, "infonce_weight", 1.0))
        init_vicreg = float(getattr(args, "vicreg_weight", 1.0))

        frac_infonce = torch.tensor(init_infonce / max(align_max, 1e-12), device=device)
        frac_vicreg = torch.tensor(init_vicreg / max(align_max, 1e-12), device=device)

        model.infonce_weight_raw.data = _logit(frac_infonce).detach()
        model.vicreg_weight_raw.data = _logit(frac_vicreg).detach()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wdecay,
    )

    prior_mu = torch.zeros(args.emb_dim).to(device)
    prior_sigma = torch.ones(args.emb_dim).to(device)
    args.prior_dist = Independent(Normal(loc=prior_mu, scale=prior_sigma), 1)
 

    warmup_ratio = getattr(args, 'warmup_ratio', 0.1)
    total_steps = args.epochs * args.steps
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    logging.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info(model)
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(split_idx['valid'])}/{len(split_idx['test'])}"
    )
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")

    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    best_train, best_valid, best_test, best_count = None, None, None, None
    best_epoch = 0
    loss_tots = []

    for epoch in range(0, args.epochs):
        loss, train_loaders = pretrain_func(
            args, model, train_loaders, aligned_data, aug_data, context_graph, optimizer, scheduler, epoch
        )
        loss_tots.append(loss)
        is_last_epoch = (epoch == args.epochs - 1)
        is_checkpoint_epoch = ((epoch + 1) % 50 == 0)

        if is_checkpoint_epoch and not is_last_epoch:
            ckpt_epoch_path = args.model_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_epoch_path)
            logger.info(f"Checkpoint saved at {ckpt_epoch_path}")

        if is_last_epoch:
            torch.save(model.state_dict(), args.model_path)
            avg_loss = sum(loss_tots) / len(loss_tots)
            logger.info(f"Finished training. Model saved at {args.model_path}. Final avg loss = {avg_loss:.4f}")

    return (
        args.model_path,
        args.dataset,
        dataset.eval_metric,
        best_train,
        best_valid,
        best_test,
        best_epoch,
        best_count,
    )


if __name__ == "__main__":
    import os

    args = get_args()

    model_tag = f"{args.model_path}"

    if hasattr(args, "note") and args.note:
        model_tag += f"-{args.note}"

    ckpt_dir = "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    args.model_path = os.path.join(ckpt_dir, model_tag + ".pt")

    logger = get_logger(__name__)
    args.logger = logger

    print(f"Model will be saved to: {args.model_path}")
    print(vars(args))

    main(args, 0)







