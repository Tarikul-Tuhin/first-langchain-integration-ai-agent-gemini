from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Weather(BaseModel):
    temperature: float
    condition: str

def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"


# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create JSON agent (no format_instructions!)
agent = create_agent(
    llm,
    tools=[weather_tool],
    response_format=Weather,
)

# Invoke agent
result = agent.invoke({"messages": [HumanMessage("What's the weather in SF?")]})

weather:Weather = result["structured_response"]

print(weather.condition)

# Parse the output string into structured object
print(repr(weather))


