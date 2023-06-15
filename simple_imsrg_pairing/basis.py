# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np


class Basis:
    def __init__(self, n_levels: int, filled_levels: int) -> None:
        self.states = np.zeros((n_levels * 2, 4), dtype=int)
        counter = 0
        for n in range(n_levels):
            for spin in [-1, 1]:
                self.states[counter][0] = counter
                self.states[counter][1] = n
                self.states[counter][2] = spin
                if n < filled_levels:
                    self.states[counter][3] = 1
                else:
                    self.states[counter][3] = 0
                counter += 1

        # Derived single arrays
        self.indices = np.array([x[0] for x in self.states])
        self.ns = np.array([x[1] for x in self.states])
        self.spins = np.array([x[2] for x in self.states])
        self.occs = np.array([x[3] for x in self.states])
        self.occbars = 1 - self.occs

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.states[i]

    def print(self, fn=None):
        fstring = "{:>5} {:>4} {:>4} {:>4}"
        header_fstring = "#{:>4} {:>4} {:>4} {:>4}"
        if fn is None:
            print(header_fstring.format("idx", "n", "s", "occ"))
            for p, n, s, occ in self:
                print(fstring.format(p, n, s, occ))
        else:
            with open(fn) as f:
                f.write(header_fstring.format("idx", "n", "s", "occ") + "\n")
                for p, n, s, occ in self:
                    print(fstring.format(p, n, s, occ) + "\n")
