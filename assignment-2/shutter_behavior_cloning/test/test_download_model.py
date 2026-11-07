#!/usr/bin/env python3
# Public tests for CPSC-4590/5590 Assignment 2 - Part II
#
# Checks that the model (and the normalization parameters, if any) that you set in
# scripts/train_utils.py can be downloaded from Google Drive with download_model_files(),
# and that the downloaded files can be loaded as the README requires: the model
# with torch.load(path, weights_only=True), and the normalization parameters with json.load().
#
# The files are downloaded to a temporary directory and deleted afterwards.
#
# NOTE: The test for the normalization params is skipped when NORM_PARAMS_URL = None.
#
# Usage:
#   colcon test --packages-select shutter_behavior_cloning --event-handlers console_direct+ \
#       --ctest-args -R test_download_model

import json
import os
import sys

import pytest

# The CMakeLists.txt puts this package's scripts directory on the PYTHONPATH for the test. It is added
# here too so that the test can also be run directly with pytest.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

try:
    import train_utils
except ImportError as e:
    pytest.fail("Failed to import train_utils.py ({}). Activate your Python virtual environment, where you "
                "installed gdown, and rebuild the package before running the test.".format(e), pytrace=False)


def check_url(name, url, hint=""):
    """
    Check that a URL in train_utils.py has been set to a Google Drive file
    :param name: name of the variable in train_utils.py
    :param url: value of the variable
    :param hint: text appended to the error message if the URL still has the placeholder
    """
    assert "<file_id>" not in url, \
        "{} in train_utils.py still has the placeholder <file_id>. Replace it with the id of your file in " \
        "Google Drive.{}".format(name, hint)
    assert "/file/d/" not in url, \
        "{} in train_utils.py is a sharing link ({}). If Google gives you a link like " \
        "https://drive.google.com/file/d/<file_id>/view?usp=sharing, set {} to " \
        "https://drive.google.com/uc?id=<file_id> instead.".format(name, url, name)


@pytest.fixture(scope="module")
def downloaded_files(tmp_path_factory):
    """
    Download the files set in train_utils.py once for all the tests below
    :return: paths of the downloaded model and normalization parameters (None if not downloaded)
    """
    output_path = tmp_path_factory.mktemp("model_files")
    return train_utils.download_model_files(output_path=str(output_path), quiet=False)


def test_urls_are_set():
    """
    Check that MODEL_URL is set, and that NORM_PARAMS_URL is either set or None
    """
    assert train_utils.MODEL_URL is not None, "MODEL_URL is not set in train_utils.py."
    check_url("MODEL_URL", train_utils.MODEL_URL)

    if train_utils.NORM_PARAMS_URL is not None:
        check_url("NORM_PARAMS_URL", train_utils.NORM_PARAMS_URL,
                  hint=" Or set NORM_PARAMS_URL = None if your model does not use normalization parameters.")


def test_download_model(downloaded_files):
    """
    Check that the model can be downloaded and loaded with torch.load(path, weights_only=True)
    """
    model_file, _ = downloaded_files
    assert model_file is not None and os.path.isfile(model_file), \
        "Failed to download the model from {}. Check that MODEL_URL in train_utils.py is correct and that " \
        "anybody with the link can download the file without logging into Google.".format(train_utils.MODEL_URL)
    assert os.path.getsize(model_file) > 0, "The model downloaded from {} is empty.".format(train_utils.MODEL_URL)

    torch = pytest.importorskip("torch", reason="PyTorch is not available to check that the model can be loaded. "
                                                "Activate your Python virtual environment and rebuild the package.")
    try:
        state_dict = torch.load(model_file, weights_only=True, map_location="cpu")
    except Exception as e:
        pytest.fail("The model downloaded from {} could not be loaded with torch.load(path, weights_only=True): "
                    "{}. Save your model with torch.save(model.state_dict(), path).".format(train_utils.MODEL_URL, e),
                    pytrace=False)
    assert isinstance(state_dict, dict) and len(state_dict) > 0, \
        "The model downloaded from {} is not a state dictionary. Save your model with " \
        "torch.save(model.state_dict(), path).".format(train_utils.MODEL_URL)


def test_download_norm_params(downloaded_files):
    """
    Check that the normalization parameters, if any, can be downloaded and loaded with json.load()
    """
    if train_utils.NORM_PARAMS_URL is None:
        pytest.skip("NORM_PARAMS_URL is None in train_utils.py: the model does not use normalization parameters.")

    _, norm_params_file = downloaded_files
    assert norm_params_file is not None and os.path.isfile(norm_params_file), \
        "Failed to download the normalization parameters from {}. Check that NORM_PARAMS_URL in train_utils.py " \
        "is correct and that anybody with the link can download the file without logging into Google, or set " \
        "NORM_PARAMS_URL = None if your model does not use normalization parameters.".format(
            train_utils.NORM_PARAMS_URL)

    try:
        with open(norm_params_file, 'r') as fid:
            json.load(fid)
    except (ValueError, UnicodeDecodeError) as e:
        pytest.fail("The normalization parameters downloaded from {} are not a JSON file ({}). Save them with "
                    "json.dump().".format(train_utils.NORM_PARAMS_URL, e), pytrace=False)
