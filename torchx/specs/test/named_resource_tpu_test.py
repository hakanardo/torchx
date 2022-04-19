# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.specs import named_resources_tpu as tpu
from torchx.specs import Resource


class NamedResourcesTest(unittest.TestCase):
    def test_tpu_v3_8(self) -> None:
        want = Resource(
            cpu=96,
            memMB=331 * 1024,
            gpu=0,
            devices={
                "cloud-tpus.google.com/v3": 8,
            },
        )
        self.assertEqual(tpu.tpu_v3_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v3_8"](), want)

    def test_tpu_v3_2048(self) -> None:
        want = Resource(
            cpu=96,
            memMB=331 * 1024,
            gpu=0,
            devices={
                "cloud-tpus.google.com/v3": 2048,
            },
        )
        self.assertEqual(tpu.tpu_v3_2048(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v3_2048"](), want)

    def test_tpu_v2_8(self) -> None:
        want = Resource(
            cpu=96,
            memMB=331 * 1024,
            gpu=0,
            devices={
                "cloud-tpus.google.com/v2": 8,
            },
        )
        self.assertEqual(tpu.tpu_v2_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v2_8"](), want)
