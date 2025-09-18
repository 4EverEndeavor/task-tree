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
    def __init__(self, task: str, context: str=None, agent_api, outputType: str):
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

        tools:
            query_code_by_summary
            query_code_by_text
            edit_code_agent
            query_project_context
            terminal_command
            web_search
            create_file
            create_test
            validate_command
            validate_statement

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
    pass

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


def add_file_to_context:
    """
    terminal based file picker with fuzzy matching and real-time keystroke updates.
    If fuzzy match maps to a directory.
    """
    pass


def add_code_to_context:
    """
    same as add_file_to_context
    first calls file picker above, but then breaks each file up into chunks. Uses vim-like bindings (j,k) to page through chunks and enter to select.
    """
    pass


def summarize(obj, max_size):
    """
    summarize this object and return a string no longer than max size.

    """
    pass

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

def generate_contextual_summary:
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

def create_code_chunk_embeddings:
    """
    splits files into chunks
    creates summaries and embeddings for each chunk
    stores summaries and embeddings in vector data base
    """
    pass

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
    An embedding is created for all of these to answer any question about project context that the agent might have, while keeping answers small in size.
    Embeddings are stored in a vector database
    
    """
    pass
def main_loop:
    """
    generate_project_context()
    initialize_ollama_api()
    option to:
        execute_task: TaskAgent(task)
        ask_question: TaskAgent(createTask(question))
        run_command:
            af: add_file_to_context()
            ac: add_code_to_context()
            q: quit
    """
    pass


