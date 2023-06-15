# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Tuple

from .basis import Basis
from .operators import Op1B, Op2B


def get_pairing_hamiltonian(basis: Basis, g: float) -> Tuple[Op1B, Op2B]:
    # 1-body
    h1 = Op1B(basis)
    for p, n_p, s_p, _ in basis:
        for q, n_q, s_q, _ in basis:
            if n_p == n_q and s_p == s_q:
                h1.mat[p][q] = n_p

    # 2-body
    h2 = Op2B(basis)
    for p, n_p, s_p, _ in basis:
        for q, n_q, s_q, _ in basis:
            if n_p != n_q or s_p == s_q:
                continue
            for r, n_r, s_r, _ in basis:
                for s, n_s, s_s, _ in basis:
                    if n_r != n_s or s_r == s_s:
                        continue
                    if s_p == s_r:
                        h2.mat[p][q][r][s] = -1 / 2 * g
                    else:
                        h2.mat[p][q][r][s] = 1 / 2 * g

    return h1, h2
