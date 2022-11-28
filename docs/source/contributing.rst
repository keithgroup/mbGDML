============
Contributing
============

We welcome all contributions, and they are greatly appreciated!
Every little bit helps, and credit will always be given.




Types of Contributions
======================

Report bugs
-----------

Report bugs at https://github.com/keithgroup/mbGDML/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.



Fix Bugs
--------

Look through the `GitHub issues <https://github.com/keithgroup/mbGDML/issues>`__ for bugs.
Anything tagged with ``bug`` and ``help wanted`` is open to whoever wants to implement it.



Implement features
------------------

Look through the `GitHub issues <https://github.com/keithgroup/mbGDML/issues>`__ for features.
Anything tagged with ``enhancement`` and ``help wanted`` is open to whoever wants to implement it.



Write Documentation
-------------------

mbGDML could always use more documentation, whether as part of the official mbGDML docs, in docstrings, or even on the web in blog posts, articles, and such.



Propose a new feature
---------------------

The best way to propose a new feature is by starting a discussion at https://github.com/keithgroup/mbGDML/discussions.

- Create a discussion in the |:bulb:| Ideas category.
- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)



Discussions
===========

If you have any questions, comments, concerns, or criticisms please start a `discussion <https://github.com/keithgroup/mbGDML/discussions>`__ so we can improve mbGDML!


*Black* style
=============

We use the `Black style <https://black.readthedocs.io/en/stable/index.html>`__ in mbGDML.
This lets you focus on writing code and hand over formatting control to *Black*.
You should periodically run *Black* when changing any Python code in reptar.
Installing *Black* can be done with ``pip install black`` and then ran with the following command while in the repository root directory.

.. code-block:: bash

    $ black ./
    All done! ✨ 🍰 ✨
    50 files left unchanged.


Get Started!
============

Ready to contribute?
Here's how to set up ``mbgdml`` for local development.

1. Fork the `mbGDML repo on GitHub <https://github.com/keithgroup/mbGDML>`__.
2. Clone your fork locally.

.. code-block:: bash

    $ git clone https://github.com/username/mbGDML
    $ cd mbGDML
    $ git remote add upstream https://github.com/keithgroup/mbGDML
    $ git fetch upstream

3. Add upstream and fetch tags.

.. code-block:: bash

    $ cd mbGDML
    $ git remote add upstream https://github.com/keithgroup/mbGDML
    $ git fetch upstream

4. Install your local copy.

.. code-block:: bash

    $ pip install .

5. Create a branch for local development.

.. code-block:: bash

    $ git checkout -b name-of-your-branch

Now you can make your changes locally.

6. Add or change any tests in ``tests/`` if fixing a bug, adding a feature, or anything else that changes source code.
We use `pytest <https://docs.pytest.org/>`__ and store any necessary files in ``tests/data/``.
Try to reuse any data already present.
If additional data is required, keep the file size as small as possible.

7. When you're done making changes, check that your changes pass the tests.

.. code-block:: bash

    $ pytest
    ======================= test session starts ========================
    platform linux -- Python 3.10.4, pytest-7.1.2, pluggy-1.0.0
    rootdir: /home/alex/repos/keith/mbGDML-dev
    plugins: anyio-3.6.1, order-1.0.1
    collected 12 items                                                 

    tests/test_datasets.py .                                     [  8%]
    tests/test_descriptors.py .                                  [ 16%]
    tests/test_mbe.py .                                          [ 25%]
    tests/test_predict.py .                                      [ 33%]
    tests/test_predictsets.py ..                                 [ 50%]
    tests/test_rdf.py .                                          [ 58%]
    tests/test_train.py .....                                    [100%]

    ======================= 12 passed in 29.55s ========================

8. Check *Black* formatting by running the ``black ./`` command.

9. Write any additional documentation in ``docs/source/``.
You can easily build and view the documentation locally by running the ``docs/branch-build-docs.sh`` script then opening ``docs/html/index.html`` in your favorite browser.

.. code-block:: bash

    $ ./docs/branch-build-docs.sh 
    Running Sphinx v5.3.0
    making output directory... done
    loading intersphinx inventory from https://urllib3.readthedocs.io/en/latest/objects.inv...
    loading intersphinx inventory from https://docs.python.org/3/objects.inv...
    loading intersphinx inventory from https://numpy.org/doc/stable/objects.inv...
    loading intersphinx inventory from https://matplotlib.org/stable/objects.inv...
    loading intersphinx inventory from https://cclib.github.io/objects.inv...
    loading intersphinx inventory from https://wiki.fysik.dtu.dk/ase/objects.inv...
    loading intersphinx inventory from https://pytorch.org/docs/master/objects.inv...
    loading intersphinx inventory from https://docs.scipy.org/doc/scipy/objects.inv...
    building [mo]: targets for 0 po files that are out of date
    building [html]: targets for 111 source files that are out of date
    updating environment: [new config] 111 added, 0 changed, 0 removed
    reading sources... [100%] training                                                                                                               
    looking for now-outdated files... none found
    pickling environment... done
    checking consistency... done
    preparing documents... done
    writing output... [100%] training                                                                                                                
    generating indices... genindex done
    highlighting module code... [100%] mbgdml.utils                                                                                                  
    writing additional pages... search done
    copying images... [100%] images/training/1h2o-cl-losses-1000-iter.png                                                                            
    copying downloadable files... [100%] files/dsets/3h2o-nbody.npz                                                                                  
    copying static files... done
    copying extra files... done
    dumping search index in English (code: en)... done
    dumping object inventory... done
    build succeeded.

    The HTML pages are in html.

10. Add a description of the changes in the ``CHANGELOG.md``.
Please follow the general format specified `here <https://keepachangelog.com/en/1.0.0/>`__.

11. Commit your changes and push your branch to GitHub.

.. code-block:: bash

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-branch

12. Submit a pull request through the `GitHub website <https://github.com/keithgroup/mbGDML>`__.




Pull Request Guidelines
=======================

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in ``CHANGELOG.md``.

.. tip::

    You can open a draft pull request first to check that GitHub actions pass for all supported Python versions.

Deploying
=========

A reminder for the maintainers on how to deploy.

Our versions are manged with `versioneer <https://github.com/python-versioneer/python-versioneer>`__.
This primarily relies on tags and distance from the most recent tag.
Creating a new version is automated with ``bump2version`` (which can be installed with ``pip install bump2version``) and controlled with ``.bumpversion.cfg``.
Then, the `Upload Python Package <https://github.com/keithgroup/mbGDML/actions/workflows/python-publish.yml>`__ GitHub Action will take care of deploying to PyPI.

.. note::

    Each push to ``main`` will trigger a TestPyPI deployment `here <https://test.pypi.org/project/mbGDML/>`__.
    Tags will trigger a PyPI deployment `here <https://pypi.org/project/mbGDML/>`__.

Create a new version of ``mbgdml`` by running the following command while in the repository root.

.. code-block:: bash

    $ bump2version patch # possible: major / minor / patch

Then, push the commit and tags.

.. code-block:: bash

    $ git push --follow-tags

