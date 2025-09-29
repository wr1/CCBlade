ruff format
ruff check --fix > out.txt
uv run pytest -v >> out.txt
git add ccblade/bem.py
git commit -m 'Implement BEM functions in Python'
git add ccblade/ccblade.py
git commit -m 'Update imports to use Python BEM'
git add pyproject.toml
pyproject.toml
git commit -m 'Update build system to remove Fortran'
git add setup.py
git commit -m 'Remove Meson extension from setup'
git add meson.build
git commit -m 'Remove Fortran from Meson'
git add admin.sh
git commit -m 'Add admin script for formatting and testing'