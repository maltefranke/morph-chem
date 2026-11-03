"""Attachment-site constraints for scaffold-decoration insertion supervision."""

import torch


def scaffold_insertion_candidates(target, n_scaffold, future_indices, orig_to_interp):
    """Eligible existing parents for each future atom, in interpolation order.

    Removing the retained scaffold from the clean target bond graph defines
    substituent branches, including explicit hydrogens. A future atom may spawn
    from its own branch or any scaffold atom bonded to that branch. Distinct
    branches cannot borrow each other's parents, even at the same attachment
    site. Multiple scaffold attachments are all retained, without an arbitrary
    choice of root. Disconnected components with no scaffold attachment have no
    site constraint and retain the global nearest-parent rule.

    The aligned scaffold occupies the first ``n_scaffold`` target slots. Token
    zero denotes ``<NO_BOND>`` by the canonical vocabulary contract. These are
    training labels derived from clean target bonds, never noisy runtime bonds;
    no target information is used during generation.
    """
    n = target.num_nodes
    if not 0 < n_scaffold <= n:
        raise ValueError("A nonempty scaffold within the target is required")
    if (orig_to_interp[:n_scaffold] < 0).any():
        raise ValueError("Every scaffold atom must exist in the interpolated state")
    if (future_indices < n_scaffold).any():
        raise ValueError("Scaffold atoms cannot be future insertions")

    adjacency = [set() for _ in range(n)]
    bonds = target.neighbor_list[target.edge_types != 0].tolist()
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)

    allowed = torch.zeros((n, n), dtype=torch.bool, device=target.device)
    unseen = set(range(n_scaffold, n))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        branch, stack, attachments = [seed], [seed], set()
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if neighbor < n_scaffold:
                    attachments.add(neighbor)
                elif neighbor in unseen:
                    unseen.remove(neighbor)
                    branch.append(neighbor)
                    stack.append(neighbor)
        rows = torch.tensor(branch, device=target.device)
        if attachments:
            parents = torch.tensor(branch + sorted(attachments), device=target.device)
            allowed[rows[:, None], parents[None, :]] = True
        else:
            allowed[rows] = True

    existing = (orig_to_interp >= 0).nonzero(as_tuple=True)[0]
    existing = existing[orig_to_interp[existing].argsort()]
    candidates = allowed[future_indices][:, existing]
    if not candidates.any(dim=1).all():
        raise ValueError("A future insertion has no eligible existing parent")
    return candidates
