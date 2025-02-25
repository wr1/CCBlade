# Project definition
project(
  'ccblade',
  'c', 'fortran',  # Include both C and Fortran languages
  license: 'Apache',
  meson_version: '>= 0.60',
  default_options: [
    'buildtype=debugoptimized',
  ],
)

# Get compilers
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')
is_windows = host_machine.system() == 'windows'

# Link math library (-lm) for C if available (safe assumption from SciPy)
m_dep = cc.find_library('m', required: false)
if m_dep.found()
  add_project_link_arguments('-lm', language: 'c')
endif

# Python setup
py_mod = import('python')
if get_option('python_target') != ''
  py = py_mod.find_installation(get_option('python_target'))
else
  py = py_mod.find_installation('python3')  # Default to python3
endif
py_dep = py.dependency()

message('Python executable: ' + py.path())
message('Python install dir: ' + py.get_install_dir())

# Build the Fortran extension module (_bem)
py.extension_module(
  '_bem',
  'ccblade/src/bem.f90',  # Source file for the Fortran extension
  dependencies: [py_dep],
  install: true,
  subdir: 'ccblade',  # Install _bem.so into site-packages/ccblade/
)

# Install Python package files
py.install_sources(
  # List all Python files explicitly
  'ccblade/__init__.py',
  'ccblade/ccblade.py',
  'ccblade/geometry.py',  # Add other .py files as needed
  subdir: 'ccblade',  # Install into site-packages/ccblade/
)

# Optional: If there’s a ccblade subdirectory with more build logic
# subdir('ccblade')  # Uncomment if additional build steps are defined there


