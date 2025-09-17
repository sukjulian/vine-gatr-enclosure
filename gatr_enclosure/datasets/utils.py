# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import re


def get_hash_id(repr_: str) -> str:

    repr_ = re.sub(r" at [0-9|a-z]*", "", repr_)  # remove instance hashes
    hash_id = hex(sum(ord(char) for char in repr_))

    return hash_id
