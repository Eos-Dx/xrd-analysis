Contributing
============

Development Workflow
--------------------

Install the `GitHub commandline interface <https://github.com/cli/cli#installation>`_ to speed up development. Make sure to follow *best practices*.

1. Create an issue, and optionally add a label

.. code-block:: console

    $ gh issue create

2. Create a branch named according to the issue type, description, and issue number.

.. code-block:: console

    $ git checkout -b feature/brilliant-new-feature/42

3. Make some changes to solve the issue, and include tests and documentation if appropriate.

4. Increase the version according to the following system: X.Y.Z, where X = major version that adds major new changes that break backwards-compatibility, Y = minor version that adds non-breaking features, Z = bug fixes or small edits.

5. Make some commits.

.. code-block:: console

   $ git add changed_file
   $ git commit -m "adds new brilliant feature"

6. Merge main to the development branch and resolve any conflicts.

.. code-block:: console

   $ git merge main

7. Create a pull request and note its number(e.g. 43).

.. code-block:: console

    $ gh pr create

8. Merge the pull request and reference the issue using the `body` tag.

.. code-block:: console

   $ gh pr merge 43 --body "Solves #42."

Note: you can answer `y` to delete the branch when merging. If you select `n` then you need to manually switch to main and pull your changes.
