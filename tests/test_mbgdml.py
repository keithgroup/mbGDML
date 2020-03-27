#!/usr/bin/env python

"""Tests for `mbgdml` package."""

import os
import pytest
import numpy as np
import mbgdml

test_path = mbgdml.utils.norm_path(
    os.path.dirname(os.path.realpath(__file__))
)

def test_data_create_dataset():
    ref_dataset_path = ''.join([test_path, 'data/ABC-4MeOH-300K-1-gdml.npz'])
    ref_output_path = ''.join([test_path, 'data/out-4MeOH-300K-1-ABC.out'])

    ref_dataset = np.load(ref_dataset_path)
    test_partition = mbgdml.data.PartitionCalcOutput(ref_output_path)
    test_partition.create_dataset()
    
    assert np.array_equal(ref_dataset.f.E, test_partition.dataset['E'])