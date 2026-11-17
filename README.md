# AI4EOSC demo agent

This is a reference implementation for users on how to define an A2A-compatible agent (or several of them) that can run either locally or within the AI4EOSC platform.

The repository provides several definitions of the same agent using different frameworks. The frameworks currently demoed are PydanticAI and Langgraph, although any frameworks compatible with the A2A protocol is supported.

All the agents are served from the same FastAPI application and use a common A2A execution layer.

Agents can be created and tested locally without registering them in LiteLLM, while still accessing language models and MCP tools through the LiteLLM gateway using an API key.


## Architecture

Framework-specific behavior is isolated behind a backend interface. Each backend converts framework-specific events into a common stream:

```python
AgentStreamEvent("thinking", ...)
AgentStreamEvent("response", ...)
AgentStreamEvent("final", ...)
```

This keeps the A2A layer independent of the underlying agent framework.


## Features

- A2A-compatible agents
- Streaming responses
- Separate thinking and response channels
- Conversation history support
- MCP tool integration
- LiteLLM gateway integration
- Shared framework-agnostic A2A executor
- Local agent hosting without LiteLLM registration
- Local testing through A2A Inspector
- CLI entry point

## Installation

Python 3.10 or newer is required.

Clone the repository and install the package:

```bash
git clone https://github.com/ai4os-hub/ai4os-demo-agent
cd agent-demo
python3 -m pip install -e .
```

## Configuration

Create a `.env` file or configure the required environment variables. The following ones are needed to run the app:

```env
LITELLM_BASE_URL=https://vllm.cloud.ai4eosc.eu
LITELLM_API_KEY=********************************************
PUBLIC_BASE_URL=http://127.0.0.1:8000
LITELLM_AGENT_MODEL=*****************************************
```
Note that a single LiteLLM API key is used by all local agents to access the authorized language models and MCP tools. No agent registration in LiteLLM is required.

The key must have the necessary permissions to access these services.

## Running

Start the server with:

```bash
ai4eosc-agent run
```

By default, it listens on:

```text
0.0.0.0:80
```

## A2A endpoints

The demo agents are exposed under separate routes on the same server:

```text
http://127.0.0.1:8000/pydanticai-demo/
http://127.0.0.1:8000/langgraph-demo/
```

## Testing with A2A Inspector

Agents can be tested locally using A2A Inspector, without deploying or registering them in LiteLLM.

1. Install A2A Inspector

Download the appropriate binary for your operating system from the A2A Inspector releases.

Extract the archive and locate the a2a-inspector executable.

On Linux, if necessary, make it executable:

```bash
chmod +x a2a-inspector
```

2. Start the local agent server, and keep that terminal running.

3. Launch A2A Inspector: Open another terminal in the directory containing the A2A Inspector executable and run

```bash
./a2a-inspector -a http://127.0.0.1:8000/langgraph-demo/.well-known/agent-card.json
```

The inspector will start a local web interface and display a URL similar to:

http://127.0.0.1:8080/?token=...

Open the complete URL, including the token, in your browser.
