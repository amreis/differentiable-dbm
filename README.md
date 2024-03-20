# Differentiable Decision Boundary Maps

This repository contains code and experiment data related to the work titled
*Exploring Classifiers with Differentiable Decision Boundary Maps*, due to A. Machado, M. Behrisch, and A. Telea, to appear in the proceedings of EuroVis 2024.

In this project, I am exploring how to augment Decision Boundary Maps
(DBMs) with information coming from Adversarial Example Generation (AEG) theory.

The AEG method implemented so far is DeepFool: Moosavi-Dezfooli, Seyed-Mohsen, Alhussein Fawzi, and Pascal Frossard. “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks.” arXiv, July 4, 2016. https://doi.org/10.48550/arXiv.1511.04599.

I also propose additional views based off of differentiating specific components in the DBM-generating process.

## How to get this thing running?

You will need:
* [Poetry](https://python-poetry.org/) installed
* Python >=3.9
* TkInter and its Python bindings. Something like `sudo apt install python3-tk`.

From the directory containing `pyproject.toml`.
```shell
poetry install
poetry run main
```