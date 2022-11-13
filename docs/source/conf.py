# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'mbGDML'
copyright = '2020-2022 Alex M. Maldonado'
author = 'Alex M. Maldonado'
html_title = 'mbGDML'





# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_multiversion',
    'sphinx_design',
    'sphinxcontrib.mermaid',
    'sphinxemoji.sphinxemoji',
]

# 
suppress_warnings = ['autosectionlabel.*']



# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Include __init__ docstring for classes
autoclass_content = 'both'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Updating master docs
master_doc = 'index'

# Add mappings
intersphinx_mapping = {
    'urllib3': ('https://urllib3.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'cclib': ('https://cclib.github.io/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
    'torch': ("https://pytorch.org/docs/master/", None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}





# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'furo'

# Including sphinx multiversion
templates_path = [
    "_templates",
]
smv_branch_whitelist = r'main'  # Only include the main branch
html_sidebars = {
    '**': [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        'versions.html',
    ],
}

# Manually copy over files to the root. These can then be referenced outside of the
# download directive.
html_extra_path = [
    './files/structures/2h2o-psi4-opt.xyz',
]