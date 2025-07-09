# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
sys.path.append('C:/Users/grego/OneDrive/Documents/3. Extracurricular/Princeton MAE Graduate School/Learning Nonlinear Projections for Dynamical Systems Using Constrained Autoencoders/romnet/src')

project = 'romnet'
copyright = '2022, Clancy Rowley, Greg Macchio, Sam Otto'
author = 'Clancy Rowley, Greg Macchio, Sam Otto'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
