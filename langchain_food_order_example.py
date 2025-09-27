from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Food(BaseModel):
    food_name: str
    food_price: float
    food_qty: int
    food_description: str

def food_order_tool(food_name: str, food_qty: int) -> Food:
    """Takes a food name and quantity, and returns the structured food order."""
    
    # Example food menu (you can replace with DB lookup or API call)
    menu = {
        "pizza": {"price": 12.5, "description": "Cheesy oven-baked pizza with toppings"},
        "burger": {"price": 8.0, "description": "Juicy grilled beef burger with lettuce & tomato"},
        "sushi": {"price": 15.0, "description": "Fresh salmon and tuna sushi rolls"},
    }

    if food_name.lower() not in menu:
        raise ValueError(f"Sorry, {food_name} is not available in the menu.")

    item = menu[food_name.lower()]
    return Food(
        food_name=food_name,
        food_price=item["price"],
        food_qty=food_qty,
        food_description=item["description"],
    )

order = food_order_tool("Pizza", 2)
print(order.model_dump())


# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create JSON agent (no format_instructions!)
agent = create_agent(
    llm,
    tools=[food_order_tool],
    response_format=Food,
)

# Invoke agent
result = agent.invoke({"messages": [HumanMessage("I want to order a food. The food name is Pizza. The food quantity is 2")]})

order:Food = result["structured_response"]

print(order.food_name)

# Parse the output string into structured object
print(repr(order))