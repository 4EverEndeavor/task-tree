"""
A terminal based python ai chat agent. Here is the overall scheme:

Integrates with ollama local agent using ollama api.

Large tasks are broken into smaller tasks and passed to more TaskAgents via tools.
When a task is passed from one agent to anther, the context is kept below a maximum size.
"""

MAX_CONTEXT_SIZE_TOKENS = 4096


class Chunk:
    def __init__(self):
        """
        An object representing a logical chunck of code.
        Keeps track of things like:
            file path
            file name
            line number start
            line number end
            summary embedding
            logical embedding
        """
        pass


"""
This is the main thing: Everything is a TaskAgent.
Any task, no matter how large can be split into smaller ones.
We need a prompt that is excellent at taking a task, and splitting it
into smaller, more manageable pieces.
Once split, it creates new TaskAgents recursively.
Each task agent splits tasks until the sub tasks can be handled by a single tool.
The returned ouput of a task is passed on to the next TaskAgent in line,
so that tasks may be executed sequentially.
"""
class TaskAgent:
    def __init__(self, task: str, context: str=None, agent_api=None, outputType: str="text"):
        """
        Args:
            task (str, required): detailed description of the task that needs to be done.
            context (str, optional): concise summary of the previous task output.
            agent_api (obj, required): interface for llm agent
            outputType (str, required): description of what this task agent should return in the 'return' tool
        """
        self.messages = []
        self.SYSTEM_PROMPT = (
            "You are a task handling code agent.\n"
            "You will be given a task with context and a set of tools.\n"
            "First, analyze the task and see if it can be completed with a single tool.\n"
            "If so, call the tool and return the output by calling the 'return' tool.\n"
            "If not, split the tasks into smaller, more manageable tasks.\n"
            "Then call the \"new_task\" tool for each sub-task.\n"
        )
        self.task = task
        self.context = context
        self.agent_api = agent_api
        self.outputType = outputType

        # tools:
        #     query_code_by_summary
        #     query_code_by_text
        #     edit_code_agent
        #     query_project_context
        #     terminal_command
        #     web_search
        #     create_file
        #     create_test
        #     validate_command
        #     validate_statement

#####################
# TOOLS
#####################


def new_task(task, context, outputType, agent_api):
    """
    Creates a new TaskAgent

    Args:
        task (str, required): detailed description of the task that needs to be done.
        context (str, optional): concise summary of the previous task output.
        agent_api (obj, required): interface for llm agent
        outputType (str, required): description of what this task agent should return in the 'return' tool

    Returns:
        str: output in the desired format from the completed task.
    """
    pass


def terminal_command(command):
    """
    Runs a command in the terminal and collects output into a file
    if output is small, returns raw output
    if output is large, summarizes content by calling "summarize" and returns
    summary.

    Args:
        command (str, required): command to be executed

    Returns:
        str: raw output or summary
    """
    pass


def refactor_method(code: Chunk):
    """
    Logically refactors the chunk into smaller logical pieces
    separating concerns into smaller, more readable methods or classes.
    Calls edit_code_chunk on the chunk to replace the contents.
    """
    pass


def refactor_file(file):
    """
    Logically splits the file into smaller, more manageable chunks.
    creates new files for different chunks
    calls create_file for any new files
    """
    pass


def create_test(chunk: Chunk):
    """
    outputs a unit test/s for given code chunk
    """
    pass


def create_file(file):
    """
    creates a new file
    """
    pass


def web_search(searchStr):
    """
    Performs a web search using the DuckDuckGo api.

    Args:
        searchStr (str, required): search string

    Returns:
        a small list of clearly summarized, relevant results.
    """
    pass


def query_project_context(query):
    """
    queries project context vector database
    returns most relevant results
    """
    # Minimal in-memory vector store based on bag-of-words cosine similarity
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    store = _GLOBAL_VECTOR_STORE
    if store is None or not store["docs"]:
        return []
    qvec = _text_vector(query)
    scored = []
    for doc in store["docs"]:
        sim = _cosine_similarity(qvec, doc["vec"]) if doc["vec"] else 0.0
        scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:5]]


def validate_command(command):
    """
    Validates any terminal command
    checks project_context.documentation
    context = {"command": command}
    return TaskAgent("Does this command conform to documentation?", context)
    """
    pass


def validate_statement(command):
    """
    validates a line of code for legitimacy
    checks project_context.documentation
    context = {"command": command}
    return TaskAgent("Does this command conform to documentation?", context)
    """
    pass


def query_code_by_summary():
    """
    searches through code summary vector database by summary
    returns most relevant code chunks
    """
    pass


def query_code_by_text():
    """
    searches through code summary vector database by text
    returns most relevant code chunks
    """
    pass


def edit_code_chunk(chunk, modification):
    """
    take the chunk and replace it with the modified chunk
    This also updates the context by generating the embeddings for any updated code
    """
    pass


#####################
# Utils
#####################


def add_file_to_context():
    """
    terminal based file picker with fuzzy matching and real-time keystroke updates.
    If fuzzy match maps to a directory.
    """
    pass


def add_code_to_context():
    """
    same as add_file_to_context
    first calls file picker above, but then breaks each file up into chunks. Uses vim-like bindings (j,k) to page through chunks and enter to select.
    """
    pass


def summarize(obj, max_size):
    """
    summarize this object and return a string no longer than max size.

    """
    text = str(obj)
    if len(text) <= max_size:
        return text
    return text[: max_size - 3] + "..."

#####################
# Context generation
#####################


def initialize_ollama_api():
    """
    The ollama server is started through the python api.
    First a selection of available models is presented.
    The user selects the model from available models.
    """
    pass


def generate_contextual_summary():
    """
    recursively calls a summarizing_agent.
    generates .all-hands/summary
    """
    pass


def index_files_and_directories():
    """
    lists all files and directories.
    Each line is given an embedding so that it may be searched later.
    """
    pass


def create_code_chunk_embeddings():
    """
    splits files into chunks
    creates summaries and embeddings for each chunk
    stores summaries and embeddings in vector data base
    """
    pass

# Simple in-memory vector store and utilities
_GLOBAL_VECTOR_STORE = None


def _text_vector(text: str):
    tokens = [t.lower() for t in text.split() if t.strip()]
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine_similarity(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def generate_project_context():
    """
    This generates a ton of information about the project including but not limited to:
        files in the directory
        create_code_chunk_embeddings
        determine_framework
        determine_language
        retrieves or generates documentation for project. I.E.
            checks python version,
            checks if a version supports a given command, etc.
        how to build or run a project
        how to test the project
    An embedding is created for all of these to answer any question about
    project context that the agent might have, while keeping answers small in size.
    Embeddings are stored in a vector database
    """
    import os
    global _GLOBAL_VECTOR_STORE

    files = []
    for root, dirs, filenames in os.walk(os.getcwd()):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for name in filenames:
            if name.startswith('.'):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                content = ""
            rel = os.path.relpath(path, os.getcwd())
            files.append({
                "path": rel,
                "content": content,
                "summary": summarize(content, 512)
            })

    docs = []
    for f in files:
        text = f"FILE: {f['path']}\n{f['summary']}"
        docs.append({
            "id": f["path"],
            "text": text,
            "vec": _text_vector(text)
        })

    _GLOBAL_VECTOR_STORE = {"docs": docs}
    print("Project context generated for", len(files), "files")
    for d in docs[:5]:
        print("-", d["id"])
    return _GLOBAL_VECTOR_STORE


def main_loop():
    """
    Interactive loop outline:
      - execute_task: TaskAgent(task)
      - ask_question: TaskAgent(createTask(question))
      - run_command:
          af: add_file_to_context()
          ac: add_code_to_context()
          q: quit
    """
    generate_project_context()
    initialize_ollama_api()
    # interactive loop would go here
