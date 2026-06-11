"""Wrappers cho ansatz của lambeq.

Hỗ trợ:
    - iqp        : IQPAnsatz (chuẩn paper Lorenz 2021)
    - sim14      : Sim14Ansatz (Sim et al. 2019)
    - sim15      : Sim15Ansatz
    - strong     : StronglyEntanglingAnsatz (PennyLane standard)
"""

from __future__ import annotations

from typing import Any


def make_ansatz(
    ansatz_name: str,
    n_layers: int,
    n_qubits_n: int = 1,
    n_qubits_s: int = 1,
) -> Any:
    """Tạo lambeq ansatz theo tên.

    Parameters
    ----------
    ansatz_name : str
        Một trong: iqp, sim14, sim15, strong.
    n_layers : int
        Số layer biến phân.
    n_qubits_n, n_qubits_s : int
        Số qubit cho AtomicType NOUN và SENTENCE.
    """
    from lambeq import AtomicType

    dims = {AtomicType.NOUN: n_qubits_n, AtomicType.SENTENCE: n_qubits_s}

    name_lower = ansatz_name.lower()
    if name_lower == "iqp":
        from lambeq import IQPAnsatz

        return IQPAnsatz(dims, n_layers=n_layers)
    elif name_lower in ("sim14", "sim_14"):
        from lambeq import Sim14Ansatz

        return Sim14Ansatz(dims, n_layers=n_layers)
    elif name_lower in ("sim15", "sim_15"):
        from lambeq import Sim15Ansatz

        return Sim15Ansatz(dims, n_layers=n_layers)
    elif name_lower in ("strong", "strongly_entangling", "stronglyentangling"):
        from lambeq import StronglyEntanglingAnsatz

        return StronglyEntanglingAnsatz(dims, n_layers=n_layers)
    else:
        raise ValueError(
            f"Unknown ansatz: {ansatz_name!r}. "
            f"Chọn: iqp, sim14, sim15, strong"
        )


def ansatz_display_name(ansatz_name: str) -> str:
    """Trả về tên đẹp để hiển thị trong report."""
    mapping = {
        "iqp": "IQP",
        "sim14": "Sim14",
        "sim15": "Sim15",
        "strong": "StronglyEntangling",
    }
    return mapping.get(ansatz_name.lower(), ansatz_name)
