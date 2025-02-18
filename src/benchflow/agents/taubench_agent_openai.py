import json
import logging
import os
from openai import OpenAI
from benchflow import BaseAgent
import requests
from typing import Any, Dict


class TauAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model = "gpt-4o"  # or any other model you prefer
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
          raise EnvironmentError("OPENAI_API_KEY env var must be set to run taubench using OpenAI's API")

    def call_api(self) -> str:
        """
        Call the OpenAI API and return the assistant's response.

        Access:
        - self.env_info: dict containing benchmark-specific data
        Returns:
            str: The assistant's content.
        """

        try:

            client = OpenAI(api_key=self.api_key)

            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": self.env_info["prompt"]}]

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            # Use a consistent key to extract the agent's response, e.g., "action"
            content = response.choices[0].message.content

            return content

        except Exception as e:
            logging.error(f"[TauAgent]: Error calling OpenAI API: {e}")
            return f"ERROR: {e}"  # Return an error message to the environment


    
    def prepare_env_info(self, state_update: Dict[str, Any]) -> Dict[str, Any]:

      #We transform environment info based on the state update
      task = state_update["task"]

      prompt = f"You are assisting the user with the following task: {task['instruction']}. " \
                 f"The tools available are: {state_update['tools']}. " \
                 f"The chat history is {state_update['history']}"

      return {"prompt": prompt}