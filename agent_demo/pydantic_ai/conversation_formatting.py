from pydantic_ai import ImageUrl
from pydantic_ai.messages import (ModelRequest, ModelResponse, UserPromptPart, TextPart)

# converts history sent by frontend to message_history format of PydanticAI
def convert_frontend_history(history: list[dict]):
    converted_messages = []

    for message in history:

        role = message.get("role")
        content = message.get("content", "")

        # if is a message sent by user
        if role == "user":

            # if it is only text
            if isinstance(content, str):

                converted_messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))

            # if it is text + image
            elif isinstance(content, list):

                user_content = []

                # gets text and/or url
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        user_content.append(part.get("text", ""))

                    elif part_type == "image_url":
                        image_data = part.get("image_url",{})

                        if isinstance(image_data, dict):
                            url = image_data.get("url")
                        else:
                            url = image_data

                        if url:
                            user_content.append(ImageUrl(url=url))

                converted_messages.append(ModelRequest(parts=[UserPromptPart(content=user_content)]))

        # if the message (just text) is sent by the agent
        elif role == "assistant":

            if isinstance(content, str):
                text = content

            elif isinstance(content, list):
                text = "\n".join(part.get("text", "")for part in content if part.get("type") == "text")

            else:
                text = str(content)

            converted_messages.append(ModelResponse(parts=[TextPart(content=text)]))

    return converted_messages