import unittest
import json
import main


class MockChatAPI:
    def __init__(self, tool_calls=None):
        self.model = "mock-model"
        self.calls = []
        self.tool_calls = tool_calls

    def chat(self, model=None, messages=None, tools=None):
        self.calls.append(("chat", model, messages, tools))
        if self.tool_calls:
            return {"message": {"tool_calls": self.tool_calls}}
        else:
            return {"message": {"content": "mock chat response"}}


class MockGenerateAPI:
    def __init__(self):
        self.model = "mock-model"
        self.calls = []

    def generate(self, model=None, prompt=None):
        self.calls.append(("generate", model, prompt))
        return {"response": "mock generate response"}


class TestTaskAgentWithMocks(unittest.TestCase):
    def test_taskagent_with_chat_api_no_tools(self):
        api = MockChatAPI()
        agent = main.TaskAgent(task="Test task", context="ctx", agent_api=api, outputType="text")
        result = agent.run()
        self.assertIn("mock chat response", result)
        self.assertTrue(api.calls)
        self.assertEqual(api.calls[0][0], "chat")
        self.assertEqual(api.calls[0][1], api.model)
        self.assertIsInstance(api.calls[0][2], list)
        self.assertGreater(len(api.calls[0][2]), 0)

    def test_taskagent_with_chat_api_with_return_tool(self):
        tool_calls = [
            {
                "id": "call_123",
                "function": {
                    "name": "return",
                    "arguments": json.dumps({"output": "final output"}),
                },
            }
        ]
        api = MockChatAPI(tool_calls=tool_calls)
        agent = main.TaskAgent(task="Test task", context="ctx", agent_api=api, outputType="text")
        result = agent.run()
        self.assertEqual(result, "final output")

    def test_taskagent_with_generate_api(self):
        api = MockGenerateAPI()
        agent = main.TaskAgent(task="Test task", context="ctx", agent_api=api, outputType="text")
        result = agent.run()
        self.assertIn("mock generate response", result)
        self.assertTrue(api.calls)
        self.assertEqual(api.calls[0][0], "generate")
        self.assertEqual(api.calls[0][1], api.model)
        self.assertIsInstance(api.calls[0][2], str)


if __name__ == "__main__":
    unittest.main()