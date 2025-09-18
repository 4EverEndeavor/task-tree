import unittest
import main


class MockChatAPI:
    def __init__(self):
        self.model = "mock-model"
        self.calls = []

    def chat(self, model=None, messages=None):
        self.calls.append(("chat", model, messages))
        return {"message": {"content": "mock chat response"}}


class MockGenerateAPI:
    def __init__(self):
        self.model = "mock-model"
        self.calls = []

    def generate(self, model=None, prompt=None):
        self.calls.append(("generate", model, prompt))
        return {"response": "mock generate response"}


class TestTaskAgentWithMocks(unittest.TestCase):
    def test_taskagent_with_chat_api(self):
        api = MockChatAPI()
        agent = main.TaskAgent(task="Test task", context="ctx", agent_api=api, outputType="text")
        result = agent.run()
        self.assertIn("mock chat response", result)
        self.assertTrue(api.calls)
        self.assertEqual(api.calls[0][0], "chat")
        self.assertEqual(api.calls[0][1], api.model)
        self.assertIsInstance(api.calls[0][2], list)
        self.assertGreater(len(api.calls[0][2]), 0)

    def test_taskagent_with_generate_api(self):
        api = MockGenerateAPI()
        agent = main.TaskAgent(task="Test task", context="ctx", agent_api=api, outputType="text")
        result = agent.run()
        self.assertIn("mock generate response", result)
        self.assertTrue(api.calls)
        self.assertEqual(api.calls[0][0], "generate")
        self.assertEqual(api.calls[0][1], api.model)
        self.assertIsInstance(api.calls[0][2], str)

    def test_taskagent_mock_fallback(self):
        agent = main.TaskAgent(task="Fallback task", context=None, agent_api=None, outputType="text")
        result = agent.run()
        self.assertIn("MOCK RESPONSE", result)
        self.assertIn("Completed task: Fallback task", result)


if __name__ == "__main__":
    unittest.main()
