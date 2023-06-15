# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Tuple

import numpy as np

from .operators import Op1B, Op2B


def make_e_denom_1b(h1: Op1B):
    energies = np.diag(h1.mat)
    dim = len(energies)

    e_denom_1b = np.zeros((dim, dim))
    for p, _, _, occ_p in h1.basis:
        if occ_p != 1:
            continue
        for q, _, _, occ_q in h1.basis:
            if occ_q != 0:
                continue
            e_denom_1b[p][q] = 1 / (energies[p] - energies[q])

    return e_denom_1b


def make_e_denom_2b(h1: Op1B):
    energies = np.diag(h1.mat)
    dim = len(energies)

    e_denom_2b = np.zeros((dim, dim, dim, dim))
    for p, _, _, occ_p in h1.basis:
        if occ_p != 1:
            continue
        for q, _, _, occ_q in h1.basis:
            if occ_q != 1:
                continue
            for r, _, _, occ_r in h1.basis:
                if occ_r != 0:
                    continue
                for s, _, _, occ_s in h1.basis:
                    if occ_s != 0:
                        continue
                    e_denom_2b[p][q][r][s] = 1 / (
                        energies[p] + energies[q] - energies[r] - energies[s]
                    )

    return e_denom_2b


def get_mbpt2_energies(h1: Op1B, h2: Op2B) -> Tuple[float, float]:
    e_denom_1b = make_e_denom_1b(h1)
    e_denom_2b = make_e_denom_2b(h1)

    e_1b = 1 * np.einsum("ia,ai,ia", e_denom_1b, h1.mat, h1.mat, optimize=True)
    e_2b = (
        1 / 4 * np.einsum("ijab,abij,ijab", e_denom_2b, h2.mat, h2.mat, optimize=True)
    )

    return e_1b, e_2b


def get_canonical_mbpt3_energies(h1: Op1B, h2: Op2B) -> Tuple[float, float, float]:
    e_denom_2b = make_e_denom_2b(h1)

    e_pp = (
        1
        / 8
        * np.einsum(
            "ijab,ijcd,ijab,abcd,cdij",
            e_denom_2b,
            e_denom_2b,
            h2.mat,
            h2.mat,
            h2.mat,
            optimize=True,
        )
    )
    e_hh = (
        1
        / 8
        * np.einsum(
            "ijab,klab,ijab,abkl,klij",
            e_denom_2b,
            e_denom_2b,
            h2.mat,
            h2.mat,
            h2.mat,
            optimize=True,
        )
    )
    e_ph = -1 * np.einsum(
        "ijab,kjac,ijab,kbic,ackj",
        e_denom_2b,
        e_denom_2b,
        h2.mat,
        h2.mat,
        h2.mat,
        optimize=True,
    )

    return e_pp, e_hh, e_ph


def get_noncanonical_mbpt3_energies(h1: Op1B, h2: Op2B) -> Tuple[float, ...]:
    e_denom_1b = make_e_denom_1b(h1)
    e_denom_2b = make_e_denom_2b(h1)

    h1_pert = h1.mat - np.diag(np.diag(h1.mat))

    e_4 = (
        1
        / 2
        * np.einsum(
            "ijab,ic,abij,abcj,ci",
            e_denom_2b,
            e_denom_1b,
            h2.mat,
            h2.mat,
            h1_pert,
            optimize=True,
        )
    )

    e_5 = (
        -1
        / 2
        * np.einsum(
            "ijab,ka,abij,ijkb,ak",
            e_denom_2b,
            e_denom_1b,
            h2.mat,
            h2.mat,
            h1_pert,
            optimize=True,
        )
    )

    e_6 = (
        -1
        / 2
        * np.einsum(
            "ijab,jkab,abij,ik,abkj",
            e_denom_2b,
            e_denom_2b,
            h2.mat,
            h1_pert,
            h2.mat,
            optimize=True,
        )
    )

    e_7 = (
        1
        / 2
        * np.einsum(
            "ijab,ijcb,abij,ac,cbij",
            e_denom_2b,
            e_denom_2b,
            h2.mat,
            h1_pert,
            h2.mat,
            optimize=True,
        )
    )

    e_8 = (
        1
        / 2
        * np.einsum(
            "ia,ijbc,ai,ajcb,cbij",
            e_denom_1b,
            e_denom_2b,
            h1_pert,
            h2.mat,
            h2.mat,
            optimize=True,
        )
    )

    e_9 = (
        1
        / 2
        * np.einsum(
            "ia,jkab,ai,ibjk,abkj",
            e_denom_1b,
            e_denom_2b,
            h1_pert,
            h2.mat,
            h2.mat,
            optimize=True,
        )
    )

    e_10 = 1 * np.einsum(
        "ijab,ia,abij,jb,ai",
        e_denom_2b,
        e_denom_1b,
        h2.mat,
        h1_pert,
        h1_pert,
        optimize=True,
    )

    e_11 = 1 * np.einsum(
        "ia,ib,ai,jb,abij",
        e_denom_1b,
        e_denom_1b,
        h1_pert,
        h1_pert,
        h2.mat,
        optimize=True,
    )

    e_12 = 1 * np.einsum(
        "ia,ijab,ai,jb,abij",
        e_denom_1b,
        e_denom_2b,
        h1_pert,
        h1_pert,
        h2.mat,
        optimize=True,
    )

    e_13 = 1 * np.einsum(
        "ia,ib,ai,ab,bi",
        e_denom_1b,
        e_denom_1b,
        h1_pert,
        h1_pert,
        h1_pert,
        optimize=True,
    )

    e_14 = -1 * np.einsum(
        "ia,ja,aj,ij,ai",
        e_denom_1b,
        e_denom_1b,
        h1_pert,
        h1_pert,
        h1_pert,
        optimize=True,
    )

    return e_4, e_5, e_6, e_7, e_8, e_9, e_10, e_11, e_12, e_13, e_14
