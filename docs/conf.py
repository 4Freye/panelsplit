import os
import sys
from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("."))
# sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "panelsplit")

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google/NumPy style docstrings
    'sphinx_autodoc_typehints',  # optional, for improved type hint handling
]


html_theme = "pydata_sphinx_theme"

# General information about the project.
project = "panelsplit"
copyright = f"2024-{datetime.now().year}, panelsplit developers"
author = "panelsplit developers"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "set_output",
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False
