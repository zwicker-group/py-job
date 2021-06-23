#!/usr/bin/env python3
"""
This example shows defining a custom model class using a decorator on a function.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import job


@job.make_model
def multiply(a, b=2):
    return a * b


# instantiate the model
model = multiply({"a": 3})
# run the instance
print(model())
