# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "OnionNet"
author = "Macabe Daley"
copyright = "2025, Macabe Daley"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # Markdown support
    "nbsphinx",  # Notebook support
    "sphinx.ext.autodoc",  # pull in docstrings
    "sphinx.ext.napoleon",  # NumPy/Google style
    "sphinx.ext.autosummary",  # generate API tables
    "sphinx_copybutton",  # “copy” buttons on code blocks
    "sphinx_autodoc_typehints",  # show Python 3 type hints
    "sphinx_tabs.tabs",  # tabbed content
    "sphinx_design",  # cards, grids, dropdowns
    "sphinx.ext.intersphinx",  # links to external docs
    "sphinxcontrib.bibtex",  # if you have a references.bib
    "sphinx.ext.mathjax",  # render math
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",  # social‐media previews
]

# Automatically create summaries for modules
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# MyST-Parser settings (for Markdown)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_heading_anchors = 6

# If you have a BibTeX file:
bibtex_bibfiles = ["references.bib"]

# Intersphinx targets
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# Allow .md files and tell Sphinx our master doc is named "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
try:
    root_doc = "index"
except NameError:
    master_doc = "index"

# If your package lives alongside docs/, make sure Sphinx can import it:
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Theme options (customize these for your repo)
html_theme_options = {
    "repository_url": "https://github.com/saezlab/onionnet",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "home_page_in_toc": False,
    "show_navbar_depth": 2,
    # "launch_buttons": {   # if you want Binder/Colab
    #   "binderhub_url": "https://mybinder.org",
    # },
}

# (Optional) if you have a logo or favicon in _static:
html_logo = "_static/.onionnet_logo_v0c.png"
html_favicon = "_static/.onionnet_logo_v0c.png"
