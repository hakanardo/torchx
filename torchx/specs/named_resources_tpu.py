# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
`torchx.specs.named_resources_tpu` contains resource definitions that represent
corresponding Google Cloud TPU VMs.

.. note::
    These resource definitions may change in future. It is expected for each user to
    manage their own resources. Follow https://pytorch.org/torchx/latest/specs.html#torchx.specs.get_named_resources
    to set up named resources.

Usage:

.. doctest::

     from torchx.specs import named_resources
     print(named_resources["tpu_v2_8"])
     print(named_resources["tpu_v3_8"])
     print(named_resources["tpu_v3_2048"])

"""

from typing import Dict, Callable, Iterable

from torchx.specs.api import Resource

GiB: int = 1024

TPU_TYPES: Iterable[str] = (
    "v2-128",
    "v2-256",
    "v2-32",
    "v2-512",
    "v2-8",
    "preemptible-v2-8",
    "v3-1024",
    "v3-128",
    "v3-2048",
    "v3-256",
    "v3-32",
    "v3-512",
    "v3-64",
    "v3-8",
    "preemptible-v3-8",
)

NAMED_RESOURCES: Dict[str, Callable[[], Resource]] = {}

def _register_type(name):
    ver, _, cores = name.rpartition("-")
    device = "cloud-tpus.google.com/" + ver
    def resource() -> Resource:
        return Resource(
            cpu=96,
            memMB=331 * GiB,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "2.6.0",
            },
            devices={
                device: int(cores),
            },
        )
    resource_name = f"tpu_{name.replace('-', '_')}"
    NAMED_RESOURCES[resource_name] = resource
    globals()[resource_name] = resource

for name in TPU_TYPES:
    _register_type(name)
