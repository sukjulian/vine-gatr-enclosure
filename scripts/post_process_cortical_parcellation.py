# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import argparse

import pyvista


def main(path: str) -> None:
    point_cloud = pyvista.read(path)

    for key in ("y", "y_model"):
        point_cloud.point_data[f"{key}_collapsed"] = (
            point_cloud.point_data[key].argmax(axis=1).astype("f4")
        )

    point_cloud.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args.path)
