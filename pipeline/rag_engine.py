'''based on the context of the retreived slack messages, make 
   a model response that answers the question. 
   so the solution should look like 
   AI answer = slack message augmented ai prompt
'''
from ollama import chat

class OllamaLLM:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate_with_context(self, query: str, recieved_data: list[str]):
        context_str = "\n\n---\n\n".join(item['text'] for item in recieved_data)
        prompt = (
            "You are a RAG AI model, you are supposed to answer based on all the slack messages you are given\n"
            f"Context:\n{context_str}\n\n"
            f"Query:\n{query}\n\n"
        )

        response = chat(
            model = self.model,
            messages = [{'role': 'user', 'content': prompt}],
            stream = False
        )

        r = response['message']['content']#print("\n", response['message']['content'])

        return r