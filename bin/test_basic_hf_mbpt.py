# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from sys import argv

from simple_imsrg_pairing.basis import Basis
from simple_imsrg_pairing.hamiltonian import get_pairing_hamiltonian
from simple_imsrg_pairing.normal_order import normal_order_trivial
from simple_imsrg_pairing.mbpt import (
    get_mbpt2_energies,
    get_canonical_mbpt3_energies,
    get_noncanonical_mbpt3_energies,
)


def print_usage_and_exit(code: int = 0):
    print(f"Usage: python3 {__file__} [num_levels] [num_filled] [g]")
    exit(code)


if len(argv) == 2 and argv[1].lower() in ["-h", "--help", "help"]:
    print_usage_and_exit()

elif len(argv) != 4:
    print_usage_and_exit(1)

num_levels = int(argv[1])
num_filled = int(argv[2])
g = float(argv[3])

basis = Basis(num_levels, num_filled)

basis.print()

h1, h2 = get_pairing_hamiltonian(basis, g)

e, f, gamma = normal_order_trivial(h1, h2)

print(f"E_HF = {e}")

mp2_es = get_mbpt2_energies(f, gamma)
mp3_es = get_canonical_mbpt3_energies(f, gamma)
mp3_es_noncanon = get_noncanonical_mbpt3_energies(f, gamma)

print(f"E_MP2 = {sum(mp2_es)} ({mp2_es})")
print(f"E_MP3 = {sum(mp3_es) + sum(mp3_es_noncanon)} ({mp3_es}, {mp3_es_noncanon})")
