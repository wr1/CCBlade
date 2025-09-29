#!/bin/bash

# Run ruff format
ruff format

# Run ruff check and fix
ruff check --fix > out.txt

# Run pytest and append to out.txt
uv run pytest -v >> out.txt

# Git add and commit for each modified file
git add pyproject.toml
ruff check pyproject.toml --fix > /dev/null 2>&1
git commit -m 'Update pyproject.toml with CLI script and subdir organization'

git add ccblade/__init__.py
ruff check ccblade/__init__.py --fix > /dev/null 2>&1
git commit -m 'Update ccblade/__init__.py for new structure'

git add ccblade/core/__init__.py
ruff check ccblade/core/__init__.py --fix > /dev/null 2>&1
git commit -m 'Add ccblade/core/__init__.py'

git add ccblade/core/airfoil.py
ruff check ccblade/core/airfoil.py --fix > /dev/null 2>&1
git commit -m 'Refactor CCAirfoil to ccblade/core/airfoil.py, remove gradient calculations'

git add ccblade/core/bem.py
ruff check ccblade/core/bem.py --fix > /dev/null 2>&1
git commit -m 'Refactor BEM functions to ccblade/core/bem.py, remove gradient calculations'

git add ccblade/core/ccblade.py
ruff check ccblade/core/ccblade.py --fix > /dev/null 2>&1
git commit -m 'Refactor CCBlade to ccblade/core/ccblade.py, remove gradient calculations'

git add ccblade/utils/__init__.py
ruff check ccblade/utils/__init__.py --fix > /dev/null 2>&1
git commit -m 'Add ccblade/utils/__init__.py'

git add ccblade/utils/airfoilprep.py
ruff check ccblade/utils/airfoilprep.py --fix > /dev/null 2>&1
git commit -m 'Refactor airfoilprep to ccblade/utils/airfoilprep.py, remove gradient calculations'

git add ccblade/utils/csystem.py
ruff check ccblade/utils/csystem.py --fix > /dev/null 2>&1
git commit -m 'Refactor csystem to ccblade/utils/csystem.py, remove gradient calculations'

git add ccblade/cli/__init__.py
ruff check ccblade/cli/__init__.py --fix > /dev/null 2>&1
git commit -m 'Add ccblade/cli/__init__.py'

git add ccblade/cli/main.py
ruff check ccblade/cli/main.py --fix > /dev/null 2>&1
git commit -m 'Create ccblade/cli/main.py for CLI functionality'

git add ccblade/ccblade_component.py
ruff check ccblade/ccblade_component.py --fix > /dev/null 2>&1
git commit -m 'Refactor ccblade_component.py, remove gradient calculations'
