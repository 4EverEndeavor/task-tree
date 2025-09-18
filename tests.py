import io
import os
import sys
import types
import unittest

import main


class TestProjectContext(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._cwd)

    def test_generate_project_context_and_query(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            store = main.generate_project_context()
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        self.assertIsInstance(store, dict)
        self.assertIn("docs", store)
        self.assertGreater(len(store["docs"]), 0)
        self.assertIn("Project context generated for", output)
        self.assertIn("tests.py", "\n".join(d["id"] for d in store["docs"]))

        results = main.query_project_context("What does main.py contain?")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        ids = [r["id"] for r in results]
        self.assertTrue(any("main.py" in i or "tests.py" in i for i in ids))


# TODO's
"""
    -Write a series of unit tests that test several flows. They should follow this overall scheme.
      
    -First initialize with generate_project_context
    -
    -- Test
         - TaskAgent("Clone a git repo, something public and well known")
         - make sure repo was cloned
    -- Test
         - TaskAgent("Tell me something about the repository.")
         - This should trigger the agent to create summaries and index and project context. It should then answer the question.
    -- Test
         - TaskAgent("Search for security vulnerabilies in this project.")
         - The agent should search all files and chunks checking for security vulnerabilities.
    -- Test
         - TaskAgent("find a bug in the project.")
         - The agent should locate a bug, suggest a fix and call for an edit. After editing, the agent should attempt to build the project or run any tests.
    -- Test
         - TaskAgent("Delete project.")
         - Project should be deleted.
    -- Test
         - TaskAgent("Search the web for the latest news stories.")
"""

if __name__ == "__main__":
    unittest.main()
