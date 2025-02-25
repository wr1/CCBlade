# Project definition
project(
  'ccblade',
  'c', 'fortran',
  license: 'Apache',
  meson_version: '>= 0.60',
  default_options: ['buildtype=debugoptimized'],
)

cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')
is_windows = host_machine.system() == 'windows'

m_dep = cc.find_library('m', required: false)
if m_dep.found()
  add_project_link_arguments('-lm', language: 'c')
endif

py_mod = import('python')
if get_option('python_target') != ''
  py = py_mod.find_installation(get_option('python_target'), pure: false)
else
  py = py_mod.find_installation('python3', pure: false)
endif
py_dep = py.dependency()

message('Python executable: ' + py.path())
message('Python install dir: ' + py.get_install_dir())

# Use f2py to build the Fortran extension
f2py = find_program('f2py')  # Ensure f2py is available
bem_so = custom_target(
  '_bem',
  output: '_bem.cpython-312-x86_64-linux-gnu.so',  # Adjust for your Python version
  input: 'ccblade/src/bem.f90',
  command: [f2py, '-c', '@INPUT@', '-m', '_bem', '--build-dir', '@OUTDIR@'],
  install: true,
  install_dir: py.get_install_dir() / 'ccblade',
)

# Install Python package files
py.install_sources(
  'ccblade/__init__.py',
  'ccblade/ccblade.py',
  'ccblade/csystem.py',
  'ccblade/ccblade_component.py',
  'ccblade/airfoilprep.py',
  subdir: 'ccblade',
)