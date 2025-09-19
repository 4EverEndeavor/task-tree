import os
import unittest

import main


class TestOllamaRealAPI(unittest.TestCase):
    def test_real_ollama_available_and_chat(self):
        # Expect initialize_ollama_api to either return a real client or raise with a clear message
        api = main.initialize_ollama_api()
        # Try a minimal call using whichever interface is present
        prompt = "Say hi"
        try:
            if hasattr(api, "chat"):
                resp = api.chat(model=getattr(api, "model", None), messages=[{"role": "user", "content": prompt}])
            else:
                resp = api.generate(model=getattr(api, "model", None), prompt=prompt)
        except Exception as e:
            self.fail(f"Ollama call failed: {e}")
        self.assertIsNotNone(resp)
        text = main.TaskAgent(task="", context=None, agent_api=api, outputType="text")._extract_text(resp)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
