# Decision Boundary Maps augmented with Adversarial Examples
(we do need a catchier name here)

In this project, I am exploring how to augment Decision Boundary Maps
(DBMs) with information coming from Adversarial Example Generation (AEG) theory.

Intuitively, we can see the distance in n-D to the closest Adv. Example as a
measure of the classifier's brittleness around that point. I'm trying to prove
that that idea makes sense and also augment DBMs in interesting ways.

The AEG method implemented so far is DeepFool: Moosavi-Dezfooli, Seyed-Mohsen, Alhussein Fawzi, and Pascal Frossard. “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks.” arXiv, July 4, 2016. https://doi.org/10.48550/arXiv.1511.04599.

More methods _might_ be implemented, but no promises.

## How to get this thing running?

You will need:
* [Poetry](https://python-poetry.org/) installed
* Python >=3.9

From the directory containing `pyproject.toml`.
```shell
poetry install
poetry run main
```