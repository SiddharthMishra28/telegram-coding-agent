# Telegram Coding Agent

A conversational coding agent deployed on Render, controlled entirely through Telegram. Built with **LangGraph**, **FastAPI**, and the **Render MCP server**.

## Features

- **Telegram-only interface** — chat with your agent via any Telegram client
- **Customizable personality** — switch between `friendly`, `professional`, `concise`, and `verbose` modes, or set a fully custom system prompt
- **OpenAI-compatible LLM** — works with any OpenAI-compatible API. Pre-configured for **NVIDIA NIM**.
- **Coding tools** — read, write, and run Python files inside an isolated per-user workspace
- **Conversation memory** — remembers context within a session
- **Render deployment** — one-click deploy with `render.yaml`

## LLM Configuration

The agent uses `ChatOpenAI` from `langchain-openai`, which works with **any OpenAI-compatible API**.

Set these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(required)* | API key for your LLM provider |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for the API |
| `LLM_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | Model identifier |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |

### Example: NVIDIA NIM

```bash
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

### Example: OpenAI

```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### Example: Local Ollama

```bash
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

## Setup

```bash
cd telegram-coding-agent
cp .env.example .env
pip install -r requirements.txt
```

Fill in `.env` with your Telegram bot token and LLM credentials.

## Run locally

```bash
python main.py
```

Set your Telegram webhook to `http://<your-ip>:8000/webhook/<TOKEN>`.

## Deploy to Render

1. Push this repo to GitHub
2. In Render Dashboard, create a new **Web Service**
3. Connect your repo and select `render.yaml`
4. Add env vars: `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
5. Deploy — your bot will be live at `https://telegram-coding-agent.onrender.com`

## Commands

- `/start` — Welcome message
- `/reset` — Clear conversation
- `/personality <mode>` — Switch agent behavior
- `/help` — Show help

## Architecture

```
Telegram → Render Webhook → FastAPI → BackgroundTask → LangGraph Agent
                                                           ↓
                                                     Tools (read/write/run)
                                                           ↓
                                                     Reply via Telegram API
```

## Learning Goals

This project demonstrates:
- **LangGraph ReAct loop** with tool calling
- **Personality injection** via dynamic system prompts
- **OpenAI-compatible LLM switching** via environment variables
- **Stateless webhook handling** with FastAPI
- **MCP-driven deployment** using Render's MCP server
