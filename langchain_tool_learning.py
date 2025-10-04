# Creating Tool
from dataclasses import dataclass
from langgraph.runtime import get_runtime
from typing import Optional
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize LLM
api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

def get_user_location() -> str:
    """Retrieve user information based on user ID."""
    runtime = get_runtime(Context)
    print(runtime, 'runTime')
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"





# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: Optional[str] = None

## Create and run the agent
agent = create_agent(
    model=llm,
    # prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    # checkpointer=checkpointer
)

result = agent.invoke({"messages": [HumanMessage("I want to order a food. The food name is Pizza. The food quantity is 2")]})

print(result['structured_response'])
