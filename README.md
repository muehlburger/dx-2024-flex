[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13879936.svg)](https://doi.org/10.5281/zenodo.13879936)

# Code for our paper

This repository contains the code for the paper *"Muehlburger, H., & Wotawa, F. (2024). FLEX: Fault Localization and Explanation with Open-Source Large Language Models in Powertrain Systems. In Proceedings of the 35th International Workshop on Principles of Diagnosis and Resilient Systems (DX 2024), Vienna, Austria, November 4-7, 2024."*

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install our used libraries.

```bash
pip install -r requirements.txt -v
```

## Usage

You find our implementation in the folder [code](./code/):

- [Preprocessing](code/0-preprocessing.ipynb)
- [Load Vectorestore Qudrant](code/1-bulk_load_to_qdrant_powertrain.py)
- [Run Experiment](code/2-run-experiment.py)
- [Format Results for Publication](code/3-generate_results_for_paper.py)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you find our work useful, please cite our paper:

```bibtex
@inproceedings{muehlburger2024flex,
  author    = {Herbert Muehlburger and Franz Wotawa},
  title     = {FLEX: Fault Localization with Open-Source LLMs in Powertrain Systems},
  booktitle = {DX 2024: Proceedings of the 35th International Workshop on Principles of Diagnosis and Resilient Systems},
  year      = {2024},
  location  = {Vienna, Austria},
  month     = {November 4-7}
}
```

## License
[Apache 2.0](LICENSE)
