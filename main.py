"""
A terminal based python ai chat agent. Here is the overall scheme:

Integrates with ollama local agent using ollama api.

Large tasks are broken into smaller tasks and passed to more TaskAgents via tools.
When a task is passed from one agent to anther, the context is kept below a maximum size.
"""
import json
import os

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


def get_tools():
    return [
        {
            "name": "new_task",
            "description": "Creates a new TaskAgent to handle a sub-task. Use this to break down a large task into smaller, more manageable pieces. When using this tool, provide a detailed description of the task, any relevant context, and the expected output type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Detailed description of the task that needs to be done."},
                    "context": {"type": "string", "description": "Concise summary of the previous task output."},
                    "outputType": {"type": "string", "description": "Description of what this task agent should return in the 'return' tool"},
                },
                "required": ["task", "outputType"],
            },
        },
        {
            "name": "terminal_command",
            "description": "Runs a command in the terminal. Use this for tasks like running scripts, managing files, or using command-line tools. The command should be a single, valid shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to be executed."},
                },
                "required": ["command"],
            },
        },
        {
            "name": "web_search",
            "description": "Performs a web search using the DuckDuckGo API. Use this to find information on the web, such as documentation, articles, or examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "create_file",
            "description": "Creates a new file with the given content. Use this to create new source files, documentation, or any other text-based file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to be created."},
                    "content": {"type": "string", "description": "The content to be written to the file."},
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "create_test",
            "description": "Creates a unit test for a given code chunk. The test should be a valid unit test for the programming language of the code chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk": {"type": "string", "description": "The code chunk to be tested."},
                },
                "required": ["chunk"],
            },
        },
        {
            "name": "query_project_context",
            "description": "Queries the project context vector database to find relevant information about the project. Use this to get information about the project's structure, dependencies, or other relevant details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search for in the project context."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "edit_code_chunk",
            "description": "Replaces a code chunk with a modified version. Use this to refactor code, fix bugs, or add new features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk": {"type": "string", "description": "The original code chunk to be replaced."},
                    "modification": {"type": "string", "description": "The modified code chunk."},
                },
                "required": ["chunk", "modification"],
            },
        },
        {
            "name": "return",
            "description": "Returns the final output of the task. Use this when the task is complete and you have the final result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output": {"type": "string", "description": "The final output of the task."},
                },
                "required": ["output"],
            },
        },
    ]


class TaskAgent:
    def __init__(self, task: str, context: str = None, agent_api=None, outputType: str = "text"):
        """Args:
            task (str, required): detailed description of the task that needs to be done.
            context (str, optional): concise summary of the previous task output.
            agent_api (obj, required): interface for llm agent
            outputType (str, required): description of what this task agent should return in the 'return' tool
        """
        self.messages = []
        self.SYSTEM_PROMPT = (
            "You are a task handling code agent.\n"
            "You will be given a task with context and a set of tools.\n"
            "Your goal is to complete the task by using the available tools.\n"
            "First, analyze the task and see if it can be completed with a single tool. If so, call the tool with the correct arguments.\n"
            "If the task is complex, break it down into smaller, more manageable sub-tasks. Then, for each sub-task, call the \"new_task\" tool to create a new TaskAgent to handle it.\n"
            "When a sub-task is complete, you will receive the output. Use this output as context for the next sub-task.\n"
            "Once all sub-tasks are complete, you will have the final result. Call the \"return\" tool to return the final output.\n"
            "Always use the tools provided. Do not attempt to answer the prompt directly.\n"
            "When you call a tool, the output will be provided to you in the next turn. You can then use this output to continue with the task.\n"
        )
        self.task = task
        self.context = context
        self.agent_api = agent_api
        self.outputType = outputType
        self.tools = get_tools()

    def run(self):
        system = {"role": "system", "content": self.SYSTEM_PROMPT}
        user = {"role": "user", "content": self._build_user_prompt()}
        self.messages = [system, user]
        while True:
            try:
                if hasattr(self.agent_api, "chat"):
                    resp = self.agent_api.chat(
                        model=getattr(self.agent_api, "model", None),
                        messages=self.messages,
                        tools=self.tools,
                    )
                    if "message" in resp and "tool_calls" in resp["message"] and resp["message"]["tool_calls"]:
                        tool_calls = resp["message"]["tool_calls"]
                        tool_outputs = []
                        for tool_call in tool_calls:
                            tool_name = tool_call["function"]["name"]
                            tool_args = tool_call["function"]["arguments"]
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                            tool_output = self._call_tool(tool_name, tool_args)
                            if tool_name == 'return':
                                return tool_output
                            tool_outputs.append({"tool_call_id": tool_call["id"], "output": tool_output})
                        self.messages.append(resp["message"])
                        self.messages.append({"role": "tool", "content": json.dumps(tool_outputs)})
                    elif "message" in resp:
                        return resp["message"].get("content") or ""
                    else:
                        return str(resp)
                elif hasattr(self.agent_api, "generate"):
                    prompt = self._build_prompt_for_generate()
                    resp = self.agent_api.generate(model=getattr(self.agent_api, "model", None), prompt=prompt)
                    return self._extract_text(resp)
            except Exception as e:
                print(e)
                return None

    def _build_user_prompt(self):
        ctx = f"\nContext:\n{self.context}\n" if self.context else ""
        return f"Task: {self.task}{ctx}\nReturn output type: {self.outputType}"

    def _build_prompt_for_generate(self):
        prompt = self.SYSTEM_PROMPT + "\n\n"
        for message in self.messages:
            prompt += f"{message['role']}: {message['content']}\n"
        prompt += "assistant:"
        return prompt

    def _extract_text(self, resp):
        if isinstance(resp, dict):
            if "message" in resp and isinstance(resp["message"], dict):
                return resp["message"].get("content") or ""
            if "response" in resp:
                return resp.get("response") or ""
        return str(resp)

    def _call_tool(self, tool_name, tool_args):
        if tool_name == "new_task":
            return new_task(
                task=tool_args.get("task"),
                context=tool_args.get("context"),
                outputType=tool_args.get("outputType"),
                agent_api=self.agent_api,
            )
        elif tool_name == "terminal_command":
            return terminal_command(tool_args.get("command"))
        elif tool_name == "web_search":
            return web_search(tool_args.get("query"))
        elif tool_name == "create_file":
            return create_file(tool_args.get("file_path"), tool_args.get("content"))
        elif tool_name == "create_test":
            return create_test(tool_args.get("chunk"))
        elif tool_name == "query_project_context":
            return query_project_context(tool_args.get("query"))
        elif tool_name == "edit_code_chunk":
            return edit_code_chunk(tool_args.get("chunk"), tool_args.get("modification"))
        elif tool_name == "return":
            return tool_args.get("output")
        else:
            return f"Unknown tool: {tool_name}"


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
    agent = TaskAgent(task=task, context=context, agent_api=agent_api, outputType=outputType)
    return agent.run()


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
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    try:
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"


def create_test(chunk: Chunk):
    """
    outputs a unit test/s for given code chunk
    """
    return "def test_placeholder():\n    assert True"


def create_file(file_path, content):
    """
    creates a new file
    """
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"File created: {file_path}"
    except Exception as e:
        return f"Error: {e}"


def web_search(query):
    """
    Performs a web search using the DuckDuckGo api.

    Args:
        query (str, required): search string

    Returns:
        a small list of clearly summarized, relevant results.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return [
        {"title": "Result 1", "summary": summarize(query, 60)},
        {"title": "Result 2", "summary": summarize(query, 60)},
    ]

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

def edit_code_chunk(chunk, modification):
    """
    take the chunk and replace it with the modified chunk
    This also updates the context by generating the embeddings for any updated code
    """
    return {"chunk": chunk, "modification": modification}


#####################
# Utils
#####################


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
    First a selection of available models is presented to the user.
    The user selects the model from available models.
    Once selected, the api object is created so that prompts can
    be sent to the ollama server.
    """
    import os
    try:
        import importlib
        ollama = importlib.import_module("ollama")
    except Exception as e:
        raise RuntimeError(f"Failed to import ollama: {e}")

    try:
        models_resp = ollama.list()
    except Exception as e:
        raise RuntimeError(f"Failed to list ollama models: {e}")

    if os.environ.get("OLLAMA_DEBUG") == "1":
        try:
            print(f"models_resp: {models_resp}")
            print(f"type(models_resp) = {type(models_resp)}")
        except Exception:
            pass

    models = []
    items = None
    items = models_resp["models"]
    if items is None:
        items = []
    for m in items:
        val = None
        if isinstance(m, dict):
            val = m.get("model") or m.get("name")
        else:
            val = getattr(m, "model", None) or getattr(m, "name", None)
        if val:
            models.append(val)
    if not models:
        raise RuntimeError("No ollama models found. Use 'ollama pull <model>' to install one.")

    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")

    while True:
        try:
            selection = input("Select a model: ")
            selected_model = models[int(selection) - 1]
            break
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

    client = getattr(ollama, "Client", None)
    if client is not None:
        try:
            api = client()
        except Exception as e:
            raise RuntimeError(f"Failed to create ollama client: {e}")
    else:
        api = ollama
    setattr(api, "model", selected_model)
    return api


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
        print("- ", d["id"])
    return _GLOBAL_VECTOR_STORE


def main_loop():
    """
    Interactive loop for the TaskAgent.
    """
    generate_project_context()
    api = initialize_ollama_api()
    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        if user_input.lower() == "switch model":
            api = initialize_ollama_api()
            messages = []
            continue

        messages.append({"role": "user", "content": user_input})

        agent = TaskAgent(task=user_input, context=None, agent_api=api, outputType="text")
        agent.messages = messages
        agent_response = agent.run()

        if agent_response:
            messages.append({"role": "assistant", "content": agent_response})
            print(f"Agent: {agent_response}")


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        pass