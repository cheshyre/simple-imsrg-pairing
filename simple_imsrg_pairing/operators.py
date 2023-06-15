# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from .basis import Basis


class Op1B:
    def __init__(self, basis: Basis, herm: str = "herm") -> None:
        self.basis = basis
        dim = len(self.basis)
        self.mat = np.zeros((dim, dim))
        self.herm = herm

    def __iadd__(self, other):
        if self.herm != other.herm or self.basis != other.basis:
            raise Exception
        self.mat += other.mat

    def __add__(self, other):
        new_op = Op1B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op += other

        return new_op

    def __isub__(self, other):
        if self.herm != other.herm or self.basis != other.basis:
            raise Exception
        self.mat -= other.mat

    def __sub__(self, other):
        new_op = Op1B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op -= other

        return new_op

    def __imul__(self, factor):
        self.mat *= factor

    def __mul__(self, factor):
        new_op = Op1B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op *= factor

        return new_op

    def __rmul__(self, factor):
        new_op = Op1B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op *= factor

        return new_op

    def is_hermitian(self) -> bool:
        return self.herm == "herm"

    def hermitize(self):
        factor = 1
        if not self.is_hermitian():
            factor = -1

        new_op = Op1B(self.basis, self.herm)

        new_op.mat += self.mat
        new_op.mat += factor * np.einsum("pq->qp", self.mat, optimize=True)
        new_op.mat *= 1 / 2

        return new_op


class Op2B:
    def __init__(self, basis: Basis, herm: str = "herm") -> None:
        self.basis = basis
        dim = len(self.basis)
        self.mat = np.zeros((dim, dim, dim, dim))
        self.herm = herm

    def __iadd__(self, other):
        if self.herm != other.herm or self.basis != other.basis:
            raise Exception
        self.mat += other.mat

    def __add__(self, other):
        new_op = Op2B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op += other

        return new_op

    def __isub__(self, other):
        if self.herm != other.herm or self.basis != other.basis:
            raise Exception
        self.mat -= other.mat

    def __sub__(self, other):
        new_op = Op2B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op -= other

        return new_op

    def __imul__(self, factor):
        self.mat *= factor

    def __mul__(self, factor):
        new_op = Op2B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op *= factor

        return new_op

    def __rmul__(self, factor):
        new_op = Op2B(self.basis, self.herm)
        new_op.mat += self.mat
        new_op *= factor

        return new_op

    def is_hermitian(self) -> bool:
        return self.herm == "herm"

    def antisymmetrize(self):
        new_op = Op2B(self.basis, self.herm)

        new_op.mat += self.mat
        new_op.mat -= np.einsum("pqrs->qprs", self.mat, optimize=True)
        new_op.mat += np.einsum("pqrs->qpsr", self.mat, optimize=True)
        new_op.mat -= np.einsum("pqrs->pqsr", self.mat, optimize=True)
        new_op.mat *= 1 / 4

        return new_op

    def hermitize(self):
        factor = 1
        if not self.is_hermitian():
            factor = -1

        new_op = Op2B(self.basis, self.herm)

        new_op.mat += self.mat
        new_op.mat += factor * np.einsum("pqrs->rspq", self.mat, optimize=True)
        new_op.mat *= 1 / 2

        return new_op
