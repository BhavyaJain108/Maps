import tiktoken
import networkx as nx
from openai import OpenAI
import os

model_name="gpt-3.5-turbo"

class InvestmentAIClient:
    TOKEN_LIMIT = 8000
    
    def __init__(self, model_name, role):
        self.api_key = os.getenv['OPENAI_API_KEY']
        self.model_name = model_name
        self.role = role
        self.messages = [{"role": "system", "content": self.role}]
        self.client = OpenAI(api_key=os.getenv['OPENAI_API_KEY'])
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.tokens = len(self.tokenizer.encode(self.messages))

    def _append_to_messages(self, message):
        self.messages.append(message)
        self.tokens += len(self.tokenizer.encode(message['content']))
        
    def _remove_from_messages(self, index):
        removedMessage = self.messages.pop(index)
        self.tokens -= len(self.tokenizer.encode(removedMessage['content']))
        
    def send_message(self, user_input):
        # Construct messages with current context and user input
        message = {"role": "user", "content": user_input}
        self._append_to_messages(message)
        
        if self.tokens > self.TOKEN_LIMIT:
            self.trim_messages(self.tokens - self.TOKEN_LIMIT)
        
        try:
            # Send request to OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            
            # Process response
            if completion and 'choices' in completion and len(completion['choices']) > 0:
                ai_response = completion.choices[0].message
                self._append_to_messages(ai_response)
                return ai_response['content']
            else:
                return "No valid AI response received."
        
        except Exception as e:
            return f"Error occurred: {str(e)}"
        
    def trim_messages(self, tokens_to_remove):
        removed_tokens = 0
        while removed_tokens < tokens_to_remove:
            self._remove_from_messages(1)
            
    