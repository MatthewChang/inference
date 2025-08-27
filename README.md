# Inference Package

This is a Python package for running inference parallelizable on slurm

## Installation

To install this package, run:

```bash
pip install .
```

## Usage

### Running Inference

You can run inference using the `run_inference` script. For example to run the exponentation example:

```bash
python -m inference.run_inference --config-path configs --config-name example.yaml
```

Configs should specify two instiantiable keys `metadata` and `processor`, which define the elements to be processed, and the code to processes them respectivly. See [inference/configs/example.yaml](inference/configs/example.yaml) for an example config which can be easily modified.


## License

This project is licensed under the MIT License.
