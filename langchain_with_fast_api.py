from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from difflib import get_close_matches
import re
from fastapi import FastAPI, HTTPException
import json
from dotenv import load_dotenv
import os

load_dotenv()


class Food(BaseModel):
    food_name: str
    food_price: float
    food_qty: int
    food_description: str

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)


# Example food menu (replace with DB/API lookup if needed)
menu = {
    "pizza": {"price": 12.5, "description": "Cheesy oven-baked pizza with toppings"},
    "burger": {"price": 8.0, "description": "Juicy grilled beef burger with lettuce & tomato"},
    "sushi": {"price": 15.0, "description": "Fresh salmon and tuna sushi rolls"},
}

def food_order_list_tool(order_text: str) -> list[Food]:
    """Parses a free-form order text and returns a list of Food items."""

    order_text = order_text.lower().strip()

    # Regex to capture "2 pizzas", "three burgers", "1 sushi"
    pattern = r"(\d+)\s*([a-zA-Z]+)"
    matches = re.findall(pattern, order_text)

    if not matches:
        raise ValueError("No valid food order found in input.")

    foods: list[Food] = []

    for qty_str, food_name in matches:
        qty = int(qty_str)

        # Fuzzy match for food name
        if food_name not in menu:
            closest = get_close_matches(food_name, menu.keys(), n=1, cutoff=0.6)
            if not closest:
                raise ValueError(f"Sorry, {food_name} is not available in the menu.")
            food_name = closest[0]

        item = menu[food_name]
        foods.append(
            Food(
                food_name=food_name,
                food_price=item["price"],
                food_qty=qty,
                food_description=item["description"],
            )
        )

    return foods

order: list[Food] = food_order_list_tool("Want to order. I want 2 pizzar and 1 burgar, 3 pizza")


api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)


class FoodList(BaseModel):
    items: list[Food]


def getFoodOrderList(prompt:str) -> str :
    # Create JSON agent (no format_instructions!)
    agent = create_agent(
        llm,
        tools=[food_order_list_tool],
        response_format=FoodList,
    )

    # Invoke agent
    results = agent.invoke({"messages": [HumanMessage(prompt)]})

    order2:list[Food] = results["structured_response"]

    return order2.items



app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/order", response_model=list[Food])
def process_order(req: PromptRequest):
    try:
        items = getFoodOrderList(req.prompt)
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))