"""Isolate PoseBusters' native energy calculation behind a wall-time limit."""

from __future__ import annotations

import multiprocessing as mp


def _serve_energy(connection):
    from posebusters.modules.energy_ratio import check_energy_ratio  # noqa: PLC0415

    connection.send("ready")
    while True:
        request = connection.recv()
        if request is None:
            break
        mol, parameters = request
        connection.send(check_energy_ratio(mol, **parameters))
    connection.close()


class EnergyRatioWorker:
    """Persistent subprocess; a timeout kills native code and resets the worker.

    A Python signal/future timeout cannot interrupt an RDKit C++ call reliably.
    Keep non-energy geometry checks in the caller so an energy timeout does not
    discard their results. The stock energy function and thresholds are used.
    """

    def __init__(self, timeout: float, *, _target=_serve_energy):
        if timeout <= 0:
            raise ValueError("Energy timeout must be positive")
        self.timeout = timeout
        self.target = _target
        self.process = None
        self.connection = None
        self.timed_out_names = []

    def _start(self):
        context = mp.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=self.target, args=(child_connection,))
        self.process.start()
        child_connection.close()
        # Imports/startup have their own allowance; they do not consume the
        # molecule's energy budget or become a false chemistry failure.
        if not self.connection.poll(120) or self.connection.recv() != "ready":
            self.close()
            raise RuntimeError("PoseBusters energy worker failed to start")

    def check(self, mol_pred, **parameters):
        """Return the stock energy result, or missing values on timeout."""
        if self.process is None:
            self._start()
        self.connection.send((mol_pred, parameters))
        if self.connection.poll(self.timeout):
            return self.connection.recv()
        name = mol_pred.GetProp("_Name") if mol_pred.HasProp("_Name") else None
        self.timed_out_names.append(name)
        print(f"PoseBusters energy timed out after {self.timeout}s: {name}", flush=True)
        self.close()
        # Match stock PoseBusters' missing-energy result. Missing checks fail
        # joint validity and remain in all-attempt denominators.
        return {
            "results": dict.fromkeys(
                (
                    "num_h_added",
                    "mol_pred_energy",
                    "ensemble_avg_energy",
                    "energy_ratio",
                    "energy_ratio_passes",
                ),
                float("nan"),
            )
        }

    def close(self):
        """Stop the subprocess and release its pipe, including after a timeout."""
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Start lazily on the first energy check."""
        return self

    def __exit__(self, *_args):
        """Always release the worker, including when another PB check raises."""
        self.close()
