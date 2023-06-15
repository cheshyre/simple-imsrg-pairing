# Copyright (c) 2023 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from sys import argv

from simple_imsrg_pairing.basis import Basis


def print_usage_and_exit(code: int = 0):
    print(f"Usage: python3 {__file__} [num_levels] [num_filled]")
    exit(code)


if len(argv) == 2 and argv[1].lower() in ["-h", "--help", "help"]:
    print_usage_and_exit()

elif len(argv) != 3:
    print_usage_and_exit(1)

num_levels = int(argv[1])
num_filled = int(argv[2])

basis = Basis(num_levels, num_filled)

basis.print()
