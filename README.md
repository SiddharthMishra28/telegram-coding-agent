# Telegram Coding Agent

A conversational coding agent deployed on Render, controlled entirely through Telegram. Built with **LangGraph**, **FastAPI**, and the **Render MCP server**.

## Features

- **Telegram-only interface** — chat with your agent via any Telegram client
- **Customizable personality** — switch between `friendly`, `professional`, `concise`, and `verbose` modes, or set a fully custom system prompt
- **Coding tools** — read, write, and run Python files inside an isolated per-user workspace
- **Conversation memory** — remembers context within a session
- **Render deployment** — one-click deploy with `render.yaml`

## Setup

```bash
cd telegram-coding-agent
cp .env.example .env
pip install -r requirements.txt
```

Fill in `.env` with your Telegram bot token and OpenAI API key.

## Run locally

```bash
python main.py
```

Set your Telegram webhook to `http://<your-ip>:8000/webhook/<TOKEN>`.

## Deploy to Render

1. Push this repo to GitHub
2. In Render Dashboard, create a new **Web Service**
3. Connect your repo and select `render.yaml`
4. Add env vars: `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`
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
- **Stateless webhook handling** with FastAPI
- **MCP-driven deployment** using Render's MCP server
