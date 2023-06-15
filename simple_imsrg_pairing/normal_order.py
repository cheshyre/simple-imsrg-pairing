# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Tuple

import numpy as np

from .operators import Op1B, Op2B


def normal_order_trivial(h1: Op1B, h2: Op2B) -> Tuple[float, Op1B, Op2B]:
    if h1.basis != h2.basis or not h1.is_hermitian() or not h2.is_hermitian():
        raise Exception

    occs = h1.basis.occs

    e_1b = np.einsum("p,pp", occs, h1.mat, optimize=True)
    e_2b = 1 / 2 * np.einsum("p,q,pqpq", occs, occs, h2.mat, optimize=True)

    f_1b = np.array(h1.mat)
    f_2b = np.einsum("r,prqr->pq", occs, h2.mat, optimize=True)

    gamma_2b = np.array(h2.mat)

    e = e_1b + e_2b
    f = f_1b + f_2b
    gamma = gamma_2b

    h1_no = Op1B(h1.basis)
    h1_no.mat = f

    h2_no = Op2B(h2.basis)
    h2_no.mat = gamma

    return e, h1_no, h2_no
