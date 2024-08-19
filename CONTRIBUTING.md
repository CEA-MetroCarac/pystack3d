# PyStack3D Contribution Guidelines

Thank you for your interest in contributing to PyStack3D. We welcome contributions from the community to improve and expand the functionality of PyStack3D.

## Code of conduct

By participating in this project, you agree to abide by the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and considerate in your interactions with others.

## How to contribute

To get an overview of the project, read the [README](README.md) file.

There are several ways you can contribute to PyStack3D, including but not limited to

* asking and answering questions in [discussions](https://github.com/CEA-MetroCarac/pystack3d/discussions),
* reporting bugs and requesting features by submitting new issues,
* adding new features and fixing bugs by creating pull requests (PRs),
* improving and maintaining consistency in the documentation by updating numpydoc-style docstrings, and
* providing reproducible examples and tutorials in Jupyter notebooks.

## Getting started

### Issues

#### Open a new issue

Before reporting a bug or requesting a feature, search to see if a related issue already exists. If the results comes up empty, you can [submit a new issue](https://github.com/CEA-MetroCarac/pystack3d/issues). Make sure you include a clear and descriptive title and provide as much detail as possible to help us understand and reproduce the issue.

> 1. What version of Python are you using? What version are the relevant libraries?
> 2. What operating system and processor architecture are you using?
> 3. What did you do?
> 4. What did you expect to see?
> 5. What did you see instead?
> 6. What was the nature of the data you were working with? (type and dimensions)

#### Solve an issue

Scan through our existing issues to find one that interests you. You can narrow down the search using the labels as filters. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make changes

To contribute to PyStack3D, you must follow the "fork and pull request" workflow below.

1. [Fork the repository.](https://github.com/CEA-MetroCarac/pystack3d/fork)
2. Clone the fork to your machine using Git and change to the directory:

       git clone https://github.com/<your-github-username>/pystack3d.git
       cd pystack3d

3. Create a new branch and check it out:

       git checkout -b <branch-name>

4. Start working on your changes! You may want to create and activate an environment, and then install all dependencies:

       python3 -m pip install poetry
       python3 -m poetry install

Remember to

* write clean and readable code by following [PEP 8](https://peps.python.org/pep-0008/) style guidelines,
* ensure docstrings adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style guidelines, and
* add Pytest-based unit tests for new features and bug fixes.

### Commit your update

When you are ready to submit your changes to GitHub, follow the steps below.

1. Ensure that your local copy of PyStack3D passes all the unit tests, including any that you may have written, using pytest.
2. Stage and commit your local files.

       git add .
       git commit -m "<short-description-of-your-changes>

3. Push changes to the `<branch-name>` branch of your GitHub fork of PyStack3D.

       git push

### Pull request

If you wish to contribute your changes to the main PyStack3D project, [make a PR](https://github.com/CEA-MetroCarac/pystack3d/compare). The project maintainers will review your PR and, if it provides a significant or useful change to PyStack3D, will be merged!