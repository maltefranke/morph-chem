import torch
import torch.nn.functional as F

from morph.model.layers.gmm import get_typed_gmm_components
from morph.utils.graph_math import (
    unsorted_segment_mean,
    unsorted_segment_mean_and_count,
)


def zero_dependency(value: torch.Tensor) -> torch.Tensor:
    """Return scalar zero while retaining ``value`` in the autograd graph."""
    if value.numel() == 0:
        return value.sum() * 0.0
    return value.reshape(-1)[0] * 0.0


def reduce_loss(loss, reduction: str = "mean"):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def _type_log_prob_from_ce(target, dist, class_weights, name, N):
    """Compute per-component type log-prob using PyTorch CE."""
    logits = dist.logits  # [N, 1, K, T]
    K = logits.shape[2]
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)

    if target.dim() == 1:
        if target.shape[0] != N:
            raise ValueError(
                f"{name} hard targets must have shape [N], got {target.shape} for N={N}"
            )
        target_idx = target.long()
        target_idx_flat = target_idx.view(N, 1, 1).expand(N, 1, K).reshape(-1)
        ce = F.cross_entropy(
            logits_flat,
            target_idx_flat,
            weight=class_weights,
            reduction="none",
        )
        return -ce.view(N, 1, K)

    if target.dim() == 2:
        if target.shape[0] != N:
            raise ValueError(
                f"{name} soft targets must have shape [N, T], "
                f"got {target.shape} for N={N}"
            )
        num_classes = dist.logits.shape[-1]
        if target.shape[1] != num_classes:
            raise ValueError(
                f"{name} soft targets class dim mismatch: "
                f"expected {num_classes}, got {target.shape[1]}"
            )

        target_probs = target.to(dtype=dist.logits.dtype)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        target_probs_flat = (
            target_probs.view(N, 1, 1, num_classes)
            .expand(N, 1, K, num_classes)
            .reshape(-1, num_classes)
        )
        ce = F.cross_entropy(
            logits_flat,
            target_probs_flat,
            weight=class_weights,
            reduction="none",
        )
        return -ce.view(N, 1, K)

    raise ValueError(
        f"{name} targets must be rank-1 indices or rank-2 "
        f"probabilities, got rank {target.dim()}"
    )


def typed_gmm_loss(
    gmm_output,
    target_x,
    target_a,
    target_c,
    class_weights_a,
    class_weights_c,
    reduction="mean",
    include_c: bool = True,
):
    """
    Computes the NLL loss using the get_typed_gmm_components helper.

    When ``include_c=False`` the joint log-probability omits the charge term
    in representations without charges. The c sub-head's outputs
    still need a DDP-safe dummy elsewhere — see the lightning module.
    """

    # Create component distributions. The helper treats the first dimension as
    # batch; here that "batch" is the valid insertion targets, so returned
    # distributions have batch_shape [N, 1, K].
    mix_dist, x_dist, a_dist, c_dist = get_typed_gmm_components(gmm_output)

    N = mix_dist.logits.shape[0]
    D = target_x.shape[-1]

    # Spatial target: [N, D] -> [N, 1, 1, D], broadcasting against K components.
    target_x = target_x.view(N, 1, 1, -1)

    # Per-component log-probabilities, all shaped [N, 1, K].
    log_prob_x = x_dist.log_prob(target_x) / D
    log_prob_a = _type_log_prob_from_ce(
        target_a, a_dist, class_weights_a, "target_a", N
    )
    # The mixture distribution already stores normalized log pi for each component.
    log_prob_mix = mix_dist.logits

    # Joint log-prob before marginalizing over mixture components. In non-charge
    # modes, omit the c term rather than training on neutral dummy targets.
    log_joint = log_prob_mix + log_prob_x + log_prob_a
    if include_c:
        log_prob_c = _type_log_prob_from_ce(
            target_c, c_dist, class_weights_c, "target_c", N
        )
        log_joint = log_joint + log_prob_c

    # Marginalize over components: log sum_k p(k, x, a[, c]).
    log_likelihood = torch.logsumexp(log_joint, dim=-1)

    if reduction == "mean":
        nll = -torch.mean(log_likelihood)
    elif reduction == "sum":
        nll = -torch.sum(log_likelihood)
    elif reduction == "none":
        nll = -log_likelihood
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    ones = torch.ones((N, 1, 1), device=target_x.device, dtype=target_x.dtype)
    return nll, (ones, ones)


def _ce_per_node(logits, target, class_weights):
    """Weighted cross-entropy per row, supporting hard (rank-1) or soft (rank-2)
    targets. Mirrors the target handling in :func:`_type_log_prob_from_ce` but
    for a single (non-mixture) distribution; returns one scalar per row.
    """
    if target.dim() == 1:
        return F.cross_entropy(
            logits, target.long(), weight=class_weights, reduction="none"
        )
    if target.dim() == 2:
        probs = target.to(dtype=logits.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return F.cross_entropy(logits, probs, weight=class_weights, reduction="none")
    raise ValueError(
        f"insertion type targets must be rank-1 indices or rank-2 probabilities, "
        f"got rank {target.dim()}"
    )


def unimodal_insertion_loss(
    uni_output,
    target_x,
    target_a,
    target_c,
    class_weights_a,
    class_weights_c,
    reduction="mean",
    include_c: bool = True,
    return_components: bool = False,
):
    """Per-node insertion loss for the unimodal (nearest-insertion) head.

    Counterpart to :func:`typed_gmm_loss` for the ``mode == "unimodal"``
    path. ``uni_output`` carries ``mu`` ``[M, 3]``, ``a_logits`` ``[M, N_a]`` and
    ``c_logits`` ``[M, N_c]``, already gathered at the ``M`` supervised nodes
    (each node's single nearest pending insertion). Position is a fixed-variance
    Gaussian (MSE), averaged over spatial dims so its scale matches the GMM
    path's per-dimension log-prob (``/D``); atom type and charge are weighted CE.

    When ``include_c=False`` the charge term is dropped (non-charge modes); the
    caller is responsible for keeping ``c_logits`` in the autograd graph for DDP.

    When ``return_components=True`` also returns a dict of the *unreduced*
    per-node component tensors (``{"pos", "type"[, "charge"]}``) so the caller can
    log them separately for diagnostics; they share the single ``ins_state`` weight.
    """
    mu = uni_output["mu"]
    a_logits = uni_output["a_logits"]
    c_logits = uni_output["c_logits"]
    D = target_x.shape[-1]

    # Fixed-variance Gaussian NLL == MSE up to constants; normalize by D so the
    # spatial term is on the same per-dimension scale as the GMM log_prob_x / D.
    loss_x = F.mse_loss(mu, target_x, reduction="none").sum(dim=-1) / D  # [M]
    loss_a = _ce_per_node(a_logits, target_a, class_weights_a)  # [M]

    loss = loss_x + loss_a
    components = {"pos": loss_x, "type": loss_a}
    if include_c:
        loss_c = _ce_per_node(c_logits, target_c, class_weights_c)
        loss = loss + loss_c
        components["charge"] = loss_c

    reduced = reduce_loss(loss, reduction)
    if return_components:
        return reduced, components
    return reduced


def do_action_loss(
    do_action_pred,
    num_actions,
    batch,
    num_graphs,
    reduction: str = "mean",
):
    """Calculate the do action loss for the given do action predictions.
    Is used for all actions (substitutions, deletions).

    Args:
        do_action_pred: The predicted do action values.
        num_actions: The number of actions for each node.
        batch: The batch indices for each node.
        num_graphs: The number of graphs in the batch.
        reduction: "mean" for a scalar or "none" for per-graph values.
    """
    do_action = num_actions > 0.0

    # NOTE: we use BCEWithLogitsLoss for scalar logit predictions
    # do_action_pred is shape (N,) with logits
    do_action_loss = F.binary_cross_entropy_with_logits(
        do_action_pred.view(-1),
        do_action.float().view(-1),
        reduction="none",
    )
    do_action_loss = unsorted_segment_mean(
        do_action_loss.view(-1, 1), batch, num_graphs
    )

    return reduce_loss(do_action_loss, reduction)


def rate_loss(rate_pred, num_actions, batch, num_graphs, reduction: str = "mean"):
    """
    Calculate the Poisson NLL rate loss for insertion predictions.

    Args:
        rate_pred: The predicted rate values, shape (N,) or (N, 1).
        num_actions: The integer number of actions for each node.
        batch: The batch indices for each node.
        num_graphs: The number of graphs in the batch.
        reduction: "mean" for a scalar or "none" for per-graph values.
    """

    # Apply Poisson NLL to ALL nodes, including those with zero insertions.
    rate_loss = F.poisson_nll_loss(
        rate_pred.view(-1),
        num_actions.view(-1),
        log_input=False,
        reduction="none",
        full=True,
    )

    # NOTE: first normalize by number of nodes / graphs
    # Otherwise, nodes with more atoms will have more weight by design
    rate_loss = unsorted_segment_mean(rate_loss.view(-1, 1), batch, num_graphs)

    return reduce_loss(rate_loss, reduction)


def class_loss(
    class_pred,
    class_target,
    class_weights,
    do_action_mask,
    batch,
    num_graphs,
    reduction: str = "mean",
):
    """
    Calculate the class loss. Is not applied to masked nodes / edges.
    Concretely:
        c: class_loss is applied to all non-del nodes.
        a: class_loss is applied to all non-del & to-be-substituted nodes.
        e: class_loss is applied to edges between non-del nodes that are substituted.
        ins_e : class_loss is applied to all edges between ins & non-del nodes.
        ins_e_ii: class_loss is applied to all edges between ins & ins nodes.
    """
    # 1. Compute CE ONLY on the nodes that need modification. Indexing with a
    # boolean mask runs an internal ``nonzero`` -- a device sync with a
    # data-dependent output size -- so derive the index ONCE and reuse it for
    # every gather instead of re-deriving it per tensor (class_pred / target /
    # batch). No empty-mask branch is needed: cross_entropy over an empty
    # selection returns an empty tensor and unsorted_segment_mean over it returns
    # zeros (count is clamped to >=1), so the loss is naturally 0 -- without the
    # host sync a Python-side ``do_action_mask.any()`` would force every call.
    idx = do_action_mask.nonzero(as_tuple=True)[0]
    masked_loss = F.cross_entropy(
        class_pred[idx],
        class_target[idx],
        weight=class_weights,
        reduction="none",
    )

    # 2. Pool the masked losses per graph.
    cls_loss, supervised_count = unsorted_segment_mean_and_count(
        masked_loss.view(-1, 1), batch[idx], num_graphs
    )

    # 3. Keep class_pred in the autograd graph unconditionally: DDP rejects
    # parameters with grad=None after backward. For a non-empty mask the gathered
    # CE already does this; the ``+ 0.0 * class_pred.sum()`` term also covers the
    # empty-mask case (zero gradient, populated) without a data-dependent branch.
    cls_loss = cls_loss + zero_dependency(class_pred)

    # 4. Determine which graphs had modifications.
    batch_has_modified = supervised_count > 0

    return reduce_loss(cls_loss, reduction), batch_has_modified
