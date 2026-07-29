# Telegram Coding Agent

A robust, plan-first coding agent deployed on Render, controlled entirely through Telegram. Built with **LangGraph**, **FastAPI**, and the **Render MCP server**.

## Workflow

1. Send any coding task to the bot
2. Bot generates a **detailed plan** and asks for approval
3. Approve / Reject with inline buttons
4. If approved, bot asks for GitHub repo name and PAT
5. Bot **incrementally creates files**, pushes to GitHub
6. Bot **monitors GitHub Actions** for build completion
7. Bot reports success/failure with logs

## Features

- **Plan-first development** with user approval at every stage
- **GitHub integration** — auto-creates repos, enables Pages, monitors Actions
- **Task queue** — only 1 task at a time, others queued
- **Persistent state** — task progress saved to disk (resumes after restart)
- **Customizable personality** — switch between `friendly`, `professional`, `concise`, `verbose`
- **OpenAI-compatible LLM** — works with any OpenAI-compatible API. Pre-configured for **NVIDIA NIM**.

## Commands

- `/start` — Welcome message
- `/reset` — Clear conversation
- `/status` — Show current task status
- `/queue` — List queued tasks
- `/cancel` — Cancel current task
- `/personality <mode>` — Switch agent behavior
- `/help` — Show help

## Setup

```bash
cd telegram-coding-agent
cp .env.example .env
pip install -r requirements.txt
```

## Run locally

```bash
python main.py
```

## Deploy to Render

1. Push this repo to GitHub
2. In Render Dashboard, create a new **Web Service**
3. Connect your repo and select `render.yaml`
4. Add env vars: `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, `GITHUB_USERNAME`, `GITHUB_PAT`
5. Deploy

## Architecture

```
Telegram → Render Webhook → FastAPI → BackgroundTasks → Task Store
                                                          ↓
                                                    Task Queue
                                                          ↓
                                                Planner → Agent → GitHub
                                                          ↓
                                                GitHub Actions Monitor
                                                          ↓
                                                    Telegram Reply
```
