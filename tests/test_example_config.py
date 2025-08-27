import pytest
from unittest import mock
import hydra
from omegaconf import OmegaConf
import inference.run_inference as run_inference

@pytest.fixture
def example_cfg():
    return OmegaConf.load("inference/configs/example.yaml")

@pytest.fixture
def processor_mock(example_cfg):
    # Instantiate metadata and processor
    metadata_handler = hydra.utils.instantiate(example_cfg.metadata)
    processor = hydra.utils.instantiate(example_cfg.processor)
    return metadata_handler, processor

@mock.patch("inference.examples.exponent.Exponent.process_results")
def test_example_config_process_results_called(mock_process_results, example_cfg):
    metadata_handler = hydra.utils.instantiate(example_cfg.metadata)
    run_inference.process_chunk(metadata_handler, example_cfg, metadata_handler.get_data())
    expected =[([2], [1]), ([4], [2]), ([8], [3]), ([16], [4]), ([32], [5]), ([64], [6]), ([128], [7]), ([256], [8]), ([512], [9])]

    for x,ex in zip(mock_process_results.call_args_list,expected):
        assert x.args == ex
