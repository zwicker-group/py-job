#!/usr/bin/env python3
"""
This example shows reading and writing data using HDF files.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from modelrunner import Result, run_function_with_cmd_args


def number_range(start: float = 1, length: int = 3):
    """create an ascending list of numbers"""
    return start + np.arange(length)


if __name__ == "__main__":
    # write result to file
    result = run_function_with_cmd_args(number_range)
    result.to_file("test.hdf")

    # write result from file
<<<<<<< Upstream, based on main
<<<<<<< Upstream, based on main
    read = Result.from_file("test.hdf")
=======
    read = Result.from_hdf("test.hdf")
>>>>>>> effedef Use State classes in rest of package
=======
    read = Result.from_file("test.hdf")
>>>>>>> 5b3d6ac More restructuring
    print(read.parameters, "–– start + [0..length-1] =", read.state)
