from dotenv import load_dotenv

loaded_env = load_dotenv()
print("Loaded API keys succesfully: ", loaded_env)

from ProductAgent import agent as product_agent
from langchain_core.messages import HumanMessage

def main():
    # query = "Create a mobile app that tracks expenses and uses AI to categorize transactions. It should connect to bank accounts, support budgets, and show charts. Make it really smooth and secure."
    query = "Create a todo application, with creating a todo, editing it and deleting it. It should have a progress bar for the todo list."

    initial_state = {
        "messages": [
            HumanMessage(content=query)
        ],
        "num_calls": 0,
        "last_review": None,
        "last_product": None,
    }

    final_state = product_agent.invoke(initial_state)

    print("\n\nnum_calls:", final_state["num_calls"])
    print("\n\nlast_review:", final_state["last_review"])
    print("\n\nlast_answer:", final_state["last_answer"])

    print("\n--- Transcript ---")
    for i, m in enumerate(final_state['messages'], 1):
        role = m.__class__.__name__
        content = getattr(m, "content", str(m))
        print(f"\n[{i}] {role}\n{content}")

    print("Saving final answer in the file")

if __name__ == "__main__":
    main()