# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import scipy.linalg

from .operators import Op1B, Op2B

from .normal_order import normal_order_trivial


class HartreeFock:
    def __init__(self, h1: Op1B, h2: Op2B) -> None:
        self.c = np.eye(len(h1.basis))
        self.h1 = 1 * h1
        self.h2 = 1 * h2
        _, f, _ = normal_order_trivial(self.h1, self.h2)
        self.last_energies = np.diag(f.mat)

    def debug_energies(self):
        print(" ".join([f"{x:>12.8f}" for x in self.last_energies]))

    def solve_iter(self) -> bool:
        e, f, _ = normal_order_trivial(self.h1, self.h2)
        print(f"E_HF = {e}")
        new_energies, c_update = scipy.linalg.eigh(f.mat)
        self.h1 = self.h1.unitary_transform(c_update)
        self.h2 = self.h2.unitary_transform(c_update)
        self.c = self.c @ c_update

        finished = np.sum(np.abs(new_energies - self.last_energies)) < 1e-6

        self.last_energies = new_energies

        return finished

    def solve(self, max_iter=100):
        for it in range(max_iter):
            if self.solve_iter():
                self.debug_energies()
                print(f"HF solved in {it + 1} iterations.")
                break
            else:
                self.debug_energies()

    def get_normal_ordered_hamiltonian(self):
        return normal_order_trivial(self.h1, self.h2)
