Contributing
============

Installation
------------

Follow :ref:`usage:Installation <usage:installation>`, replacing the final step as follows.

.. code-block:: console

    (eos) $ pip install -e .


Development Workflow
--------------------

Use the GitHub commandline interface to speed up development. Make sure to follow **best practices**.

1. Create an issue, and optionally add a label

.. code-block:: console

    $ gh issue create

2. Create a branch named according to the issue type, description, and issue number.

.. code-block:: console

    $ git checkout -b feature/brilliant-new-feature/42

3. Solve the issue and include tests and documentation if appropriate.

4. Make some commits.

.. code-block:: console

   $ git add changed_file
   $ git commit -m "adds new brilliant feature"

5. Merge main to the development branch and resolve any conflicts.

.. code-block:: console

   $ git merge main

6. Create a pull request and note its number(e.g. 43).

.. code-block:: console

    $ gh pr create

7. Merge the pull request and reference the issue.

.. code-block:: console

   $ gh pr merge 43 --body "Solves #42."
