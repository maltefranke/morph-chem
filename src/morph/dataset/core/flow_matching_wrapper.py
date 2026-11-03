import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from morph.dataset.core.atomic_data import chembatch_from_graphs, iter_geometry
from morph.dataset.core.insertion_targets import collate_insertion_targets
from morph.dataset.core.representation import (
    Representation,
    project_molecule_to_representation,
    validate_representation,
)
from morph.dataset.core.vocab import Vocab
from morph.flow_matching.prior import sample_prior_graph


def uniform_n_atoms_range(distributions, padding: int = 5) -> tuple[int, int]:
    """Return the uniform source-count bounds around the empirical support."""
    padding = int(padding)
    if padding < 0:
        raise ValueError(f"uniform n-atoms padding must be non-negative, got {padding}")
    support = (distributions.n_atoms_distribution > 0).nonzero(as_tuple=True)[0]
    return (
        max(int(support[0].item()) - padding, 3),
        int(support[-1].item()) + padding,
    )


class FlowMatchingDatasetWrapper(Dataset):
    """Create interpolated training items inside DataLoader workers.

    Training returns ``(mol_t, mol_1, insertion_targets, t)``; validation and
    test return a sampled prior and target pair. Multiple augmentations share
    the same interpolation and differ only by rotation.
    """

    def __init__(
        self,
        base_dataset,
        distributions,
        interpolator,
        vocab: Vocab,
        representation: str | Representation = Representation.MOLECULE,
        n_atoms_strategy="uniform",
        uniform_n_atoms_padding: int = 5,
        time_dist=None,
        stage="train",
        rotate: bool = True,
        n_augmentations: int = 1,
        size_sampler=None,
    ):
        self.representation = Representation(representation)
        caps = getattr(type(base_dataset), "CAPABILITIES", None)
        if caps is None:
            raise ValueError(
                f"Dataset {type(base_dataset).__name__} must declare a "
                f"CAPABILITIES attribute to be used with a representation."
            )
        validate_representation(caps, self.representation)

        self.base_dataset = base_dataset
        self.distributions = distributions
        self.interpolator = interpolator
        self.vocab = vocab
        self.n_atoms_strategy = n_atoms_strategy
        self.uniform_n_atoms_padding = int(uniform_n_atoms_padding)
        # An optional learned sampler can travel with a checkpoint and be attached
        # post-construction instead of re-deriving the starting-size rule here.
        self.size_sampler = size_sampler
        self.time_dist = (
            torch.distributions.Uniform(0.0, 1.0) if time_dist is None else time_dist
        )
        self.stage = stage
        self.rotate = rotate
        self.n_augmentations = max(1, int(n_augmentations))

        if self.n_atoms_strategy == "uniform":
            self._n_atoms_min, self._n_atoms_max = uniform_n_atoms_range(
                self.distributions, self.uniform_n_atoms_padding
            )

    def _get_n_atoms(self, target):
        if self.n_atoms_strategy == "approx":
            n = target.num_nodes + int((torch.randn(1) * 2).round().item())
            return max(3, n)
        if self.n_atoms_strategy == "uniform":
            return int(
                torch.randint(self._n_atoms_min, self._n_atoms_max + 1, (1,)).item()
            )
        if self.n_atoms_strategy == "learned":
            if self.size_sampler is None:
                raise ValueError(
                    'n_atoms_strategy="learned" requires a trained `size_sampler`'
                )
            return int(self.size_sampler.sample(1).item())
        if self.n_atoms_strategy == "empirical":
            return None
        raise ValueError(
            f"unknown n_atoms_strategy {self.n_atoms_strategy!r}; expected one "
            "of 'uniform', 'approx', 'empirical', 'learned'"
        )

    def _rotate(self, *molecules):
        """Apply one shared random rotation to a group of molecular states."""
        if self.rotate:
            R = torch.from_numpy(Rotation.random(1).as_matrix()[0]).to(
                dtype=molecules[0].positions.dtype
            )
            for mol in molecules:
                for graph in iter_geometry(mol):
                    graph.positions = graph.positions @ R

    def _augment(self, mol_t, mol_1, ins_targets, t):
        """Return one copy with a shared random rotation across all targets."""
        augmented = tuple(mol.clone() for mol in (mol_t, mol_1, ins_targets))
        self._rotate(*augmented)
        return *augmented, t

    def _training_output(self, mol_t, mol_1, ins_targets, t):
        """Apply configured rotations and return one or several training items."""
        if self.n_augmentations == 1:
            self._rotate(mol_t, mol_1, ins_targets)
            return mol_t, mol_1, ins_targets, t
        return [
            self._augment(mol_t, mol_1, ins_targets, t)
            for _ in range(self.n_augmentations)
        ]

    def _interpolate_train_atomic(
        self,
        source_atomic,
        target,
        t,
        scaffold_pairs=None,
        existing_noise="move",
    ):
        """Interpolate a training pair into native Morph graph objects.

        Insertion targets retain their structured bond records. The collator
        offsets their current-node and inserted-node indices independently.
        """
        interp, target_state, ins = self.interpolator.interpolate_single(
            source_atomic,
            target,
            t.item(),
            scaffold_pairs=scaffold_pairs,
            existing_noise=existing_noise,
        )
        return interp, target_state, ins

    def __getitem__(self, index):
        """Build one interpolated source/target training example."""
        target = self.base_dataset[index]
        target = project_molecule_to_representation(
            target, self.vocab, self.representation
        )

        n_atoms = self._get_n_atoms(target)

        if self.stage == "train":
            t = self.time_dist.sample((1,)).squeeze(0)
            t = torch.clamp(t, min=0.0, max=1 - 1e-8)

            mol_t, mol_1, ins_targets = self._interpolate_train_atomic(
                sample_prior_graph(self.distributions, n_atoms=n_atoms), target, t
            )

            if hasattr(target, "y") and target.y is not None:
                mol_t.add_system_property("y", target.y)

            self._attach_context(index, mol_t)

            return self._training_output(mol_t, mol_1, ins_targets, t)

        sample = sample_prior_graph(self.distributions, n_atoms=n_atoms)
        self._attach_context(index, sample)
        return sample, target

    def _attach_context(self, index, mol_t):
        """Task hook for fixed geometric context, attached before augmentation."""

    def __len__(self):
        """Return the number of examples in the wrapped dataset."""
        return len(self.base_dataset)


def train_collate_fn(batch, *, context_names=()):
    """Batch molecular states and structured insertion supervision."""
    samples = [
        sample
        for item in batch
        for sample in (item if isinstance(item, list) else [item])
    ]
    mols_t, mols_1, insertions, times = zip(*samples, strict=True)
    return (
        chembatch_from_graphs(list(mols_t), context_names=context_names),
        chembatch_from_graphs(list(mols_1)),
        collate_insertion_targets(list(insertions), [mol.num_nodes for mol in mols_t]),
        torch.stack(times),
    )


def eval_collate_fn(batch, *, context_names=()):
    """Collate prior/target pairs into native ``ChemBatch`` objects."""
    samples, targets = zip(*batch, strict=True)
    return (
        chembatch_from_graphs(list(samples), context_names=context_names),
        chembatch_from_graphs(list(targets)),
    )


def worker_init_fn(_worker_id):
    """Seed numpy per worker so scipy/numpy randomness diverges across workers."""
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2**32)
    np.random.seed(seed)
