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
            "Use the following context to answer the query, you are a RAG model, however don't mention slack messages.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Query:\n{query}\n\n"
            "Answer:"
        )

        response = chat(
            model = self.model,
            messages = [{'role': 'user', 'content': prompt}],
            stream = False
        )

        print(response['message']['content'])
        return