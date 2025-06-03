import os
from crewai import LLM
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini/gemini-2.0-flash")
print(f"Loaded models:\n\tOpenAI:\t{OPENAI_MODEL_NAME}\n\tGoogle:\t{GOOGLE_MODEL_NAME}")

class Dog(BaseModel):
    name: str
    age: int
    breed: str


#llm = LLM(model="gpt-4o", response_format=Dog)
llm = LLM(model=OPENAI_MODEL_NAME, response_format=Dog)

response = llm.call(
    "Analyze the following messages and return the name, age, and breed. "
    "Meet Kona! She is 3 years old and is a black german shepherd."
)
print(response)
print(llm.model)

# Output:
# Dog(name='Kona', age=3, breed='black german shepherd')