Write a series of unit tests that test several flows. They should follow this overall scheme.
 
First initialize with generate_project_context

- Test
    - TaskAgent("Clone a git repo, something public and well known")
    - make sure repo was cloned
- Test
    - TaskAgent("Tell me something about the repository.")
    - This should trigger the agent to create summaries and index and project context. It should then answer the question.
- Test
    - TaskAgent("Search for security vulnerabilies in this project.")
    - The agent should search all files and chunks checking for security vulnerabilities.
- Test
    - TaskAgent("find a bug in the project.")
    - The agent should locate a bug, suggest a fix and call for an edit. After editing, the agent should attempt to build the project or run any tests.
- Test
    - TaskAgent("Delete project.")
    - Project should be deleted.
- Test
    - TaskAgent("Search the web for the latest news stories.")
