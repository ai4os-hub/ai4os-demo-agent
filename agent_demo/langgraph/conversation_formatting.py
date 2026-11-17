from langchain_core.messages import HumanMessage, AIMessage


def convert_frontend_history(raw_history: list[dict]):
    messages = []

    for item in raw_history:
        role = item.get("role")
        content = item.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages