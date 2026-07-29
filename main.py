import logging
import os
import re
import sys
import time
import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from task_state import TaskStore, TaskState, Task
from github_manager import GitHubManager
from planner import generate_plan, revise_plan
from agent import create_agent
from personality import PersonalityConfig
from langchain_core.messages import HumanMessage
from keyboards import plan_approval_keyboard, task_control_keyboard

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("telegram-bot")

app = FastAPI()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
store = TaskStore()
personality = PersonalityConfig.from_env()
conversations: dict[int, list] = {}


def _tg(method: str, payload: dict) -> dict:
    with httpx.Client() as client:
        resp = client.post(f"{BASE_URL}/{method}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


def send_message(chat_id: int, text: str, reply_markup: dict | None = None):
    logger.info("Sending message to %s: %s", chat_id, text[:200])
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    _tg("sendMessage", payload)


def answer_callback(callback_query_id: str, text: str | None = None, show_alert: bool = False):
    payload = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    if show_alert:
        payload["show_alert"] = show_alert
    _tg("answerCallbackQuery", payload)


def send_typing(chat_id: int):
    _tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})


GITHUB_PAGES_WORKFLOW = """name: Deploy to GitHub Pages
on:
  push:
    branches:
      - main
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./out
  deploy:
    environment:
      name: github-pages
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/deploy-pages@v4
"""


def process_task(task: Task):
    try:
        if task.state == TaskState.planning:
            _handle_planning(task)
        elif task.state == TaskState.executing:
            _handle_executing(task)
        elif task.state == TaskState.pushing:
            _handle_pushing(task)
        elif task.state == TaskState.monitoring_actions:
            _handle_monitoring(task)
    except Exception as e:
        logger.exception("Task %s failed: %s", task.task_id, e)
        store.fail_active(str(e))
        send_message(task.chat_id, f"⚠️ Task failed: `{e}`")


def _handle_planning(task: Task):
    send_typing(task.chat_id)
    if task.plan and task.attempts > 0:
        plan = revise_plan(task.prompt, task.plan)
    else:
        plan = generate_plan(task.prompt)
    store.update_task(task.task_id, plan=plan, state=TaskState.awaiting_approval, attempts=task.attempts + 1)
    send_message(
        task.chat_id,
        f"📋 *Plan (attempt {task.attempts + 1}/{task.max_attempts})*\n\n{plan}\n\nDo you approve this plan?",
        reply_markup=plan_approval_keyboard(task.task_id),
    )


def _handle_executing(task: Task):
    send_typing(task.chat_id)
    workspace_dir = os.path.join(os.getcwd(), "workspace", str(task.user_id), task.task_id)
    os.makedirs(workspace_dir, exist_ok=True)

    prompt = (
        f"Create the following project according to this plan:\n\n{task.plan}\n\n"
        "Use write_file to create every required file. After creating all files, provide a summary."
    )
    agent = create_agent(personality.get_system_prompt(), workspace_dir)
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 20},
    )

    files = []
    for root, _, filenames in os.walk(workspace_dir):
        for fname in filenames:
            full = os.path.join(root, fname)
            files.append(os.path.relpath(full, workspace_dir).replace("\\", "/"))

    store.update_task(task.task_id, files=files, state=TaskState.pushing)
    send_message(task.chat_id, f"✅ Created {len(files)} files locally.\n\nPushing to GitHub...")


def _handle_pushing(task: Task):
    gh = GitHubManager(task.github_pat, task.github_username)
    workspace_dir = os.path.join(os.getcwd(), "workspace", str(task.user_id), task.task_id)

    try:
        repo = gh.create_repo(task.repo_name)
        gh.ensure_pages(task.repo_name, task.branch)
        gh.create_workflow(task.repo_name, GITHUB_PAGES_WORKFLOW)

        for file_path in task.files:
            full_path = os.path.join(workspace_dir, file_path)
            if not os.path.exists(full_path):
                continue
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            existing = gh.get_file(task.repo_name, file_path, task.branch)
            if existing:
                gh.update_file(task.repo_name, file_path, content, f"Update {file_path}", existing["sha"], task.branch)
            else:
                gh.create_file(task.repo_name, file_path, content, f"Add {file_path}", task.branch)

        store.update_task(task.task_id, state=TaskState.monitoring_actions)
        pages_url = f"https://{task.github_username}.github.io/{task.repo_name}/"
        store.update_task(task.task_id, pages_url=pages_url)
        send_message(task.chat_id, f"🚀 Pushed to GitHub. Monitoring Actions build...\n\nPages: {pages_url}")
        _handle_monitoring(task)
    except Exception as e:
        logger.exception("Push failed for task %s", task.task_id)
        store.fail_active(str(e))
        send_message(task.chat_id, f"⚠️ Push failed: `{e}`")


def _handle_monitoring(task: Task):
    gh = GitHubManager(task.github_pat, task.github_username)
    try:
        result = gh.wait_for_run(task.repo_name, timeout=300, poll_interval=15)
        if result.get("conclusion") == "success":
            store.complete_active()
            send_message(
                task.chat_id,
                f"🎉 *Build succeeded!*\n\nPages: {task.pages_url}\n\nRepo: https://github.com/{task.github_username}/{task.repo_name}",
                reply_markup=task_control_keyboard(task.task_id),
            )
        else:
            logs = result.get("logs", "")[-2000:]
            store.fail_active(result.get("conclusion", "failed"))
            send_message(
                task.chat_id,
                f"❌ Build failed with conclusion: {result.get('conclusion')}\n\nLogs:\n```\n{logs}\n```",
                reply_markup=task_control_keyboard(task.task_id),
            )
    except TimeoutError as e:
        store.fail_active(str(e))
        send_message(task.chat_id, f"⏰ Build timed out: `{e}`")
    except Exception as e:
        logger.exception("Monitoring failed for task %s", task.task_id)
        store.fail_active(str(e))
        send_message(task.chat_id, f"⚠️ Monitoring failed: `{e}`")


@app.post("/webhook/{token}")
async def webhook(token: str, request: Request, background_tasks: BackgroundTasks):
    if token != BOT_TOKEN:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    update = await request.json()

    callback = update.get("callback_query")
    if callback:
        background_tasks.add_task(handle_callback, callback)
        return JSONResponse({"ok": True})

    message = update.get("message")
    if not message or "text" not in message:
        return JSONResponse({"ok": True})

    user_id = message["from"]["id"]
    chat_id = message["chat"]["id"]
    text = message["text"].strip()
    logger.info("Received message from %s: %s", user_id, text[:200])

    if text.startswith("/"):
        background_tasks.add_task(handle_command, user_id, chat_id, text)
    else:
        background_tasks.add_task(handle_message, user_id, chat_id, text)

    return JSONResponse({"ok": True})


def handle_callback(callback: dict):
    user_id = callback["from"]["id"]
    chat_id = callback["chat"]["id"]
    data = callback["data"]
    callback_query_id = callback["id"]

    task = store.get_active_task()
    if not task or task.user_id != user_id:
        answer_callback(callback_query_id, "No active task", show_alert=True)
        return

    if data.startswith("plan:"):
        _, action, task_id = data.split(":", 2)
        if task_id != task.task_id:
            answer_callback(callback_query_id, "Task mismatch", show_alert=True)
            return
        if action == "approve":
            store.update_task(task.task_id, state=TaskState.awaiting_repo)
            answer_callback(callback_query_id, "Plan approved")
            send_message(chat_id, "✅ Plan approved. Provide a GitHub repo name (e.g. `my-calculator-app`).")
        elif action == "reject":
            answer_callback(callback_query_id, "Plan rejected")
            store.update_task(task.task_id, state=TaskState.planning)
            send_message(chat_id, "What should I change in the plan?")

    elif data.startswith("task:"):
        _, action, task_id = data.split(":", 2)
        if task_id != task.task_id:
            answer_callback(callback_query_id, "Task mismatch", show_alert=True)
            return
        if action == "cancel":
            store.cancel_active()
            answer_callback(callback_query_id, "Task cancelled")
            send_message(chat_id, "🛑 Task cancelled.")
        elif action == "pause":
            store.update_task(task.task_id, state=TaskState.queued)
            answer_callback(callback_query_id, "Task paused")
            send_message(chat_id, "⏸ Task paused.")
        elif action == "resume":
            store.set_active(task.task_id)
            answer_callback(callback_query_id, "Task resumed")
            send_message(chat_id, "▶ Task resumed.")


def handle_command(user_id: int, chat_id: int, text: str):
    cmd = text.split(maxsplit=1)[0].lower()

    if cmd == "/start":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_welcome_message())
        return

    if cmd == "/reset":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_reset_message())
        return

    if cmd == "/help":
        send_message(chat_id, personality.get_help_message())
        return

    if cmd == "/status":
        task = store.get_active_task()
        if not task:
            send_message(chat_id, "No active task.")
            return
        send_message(chat_id, f"Task `{task.task_id}`\nState: {task.state.value}\nPrompt: {task.prompt[:200]}")
        return

    if cmd == "/queue":
        tasks = store.get_user_tasks(user_id)
        if not tasks:
            send_message(chat_id, "No tasks in queue.")
            return
        lines = [f"`{t.task_id}` - {t.state.value} - {t.prompt[:100]}" for t in tasks]
        send_message(chat_id, "📋 Your tasks:\n" + "\n".join(lines))
        return

    if cmd == "/cancel":
        task = store.get_active_task()
        if not task or task.user_id != user_id:
            send_message(chat_id, "No active task to cancel.")
            return
        store.cancel_active()
        send_message(chat_id, "🛑 Task cancelled.")
        return

    if cmd == "/personality":
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            send_message(chat_id, "Usage: `/personality <friendly|professional|concise|verbose|custom>`")
            return
        tone = parts[1].strip().lower()
        valid = {"friendly", "professional", "concise", "verbose", "custom"}
        if tone not in valid:
            send_message(chat_id, f"Unknown tone. Choose from: {', '.join(sorted(valid))}")
            return
        if tone == "custom":
            send_message(chat_id, "Set `AGENT_CUSTOM_PERSONA` in your Render env vars.")
            return
        personality.tone = tone
        send_message(chat_id, f"Personality updated to *{tone}*.")
        return

    send_message(chat_id, "Unknown command. Use /help to see available commands.")


def handle_message(user_id: int, chat_id: int, text: str):
    text = text.strip()
    task = store.get_active_task()

    if task and task.user_id == user_id:
        if task.state == TaskState.awaiting_repo:
            repo_name = re.sub(r"[^a-zA-Z0-9\-]", "-", text.strip().lower())
            store.update_task(task.task_id, repo_name=repo_name, state=TaskState.awaiting_pat)
            send_message(chat_id, f"Repo name set to `{repo_name}`.\n\nPlease provide your GitHub PAT.")
            return
        elif task.state == TaskState.awaiting_pat:
            token = re.search(r"([A-Za-z0-9_]{30,})", text)
            if not token:
                send_message(chat_id, "Please provide a valid GitHub PAT.")
                return
            store.update_task(task.task_id, github_pat=token.group(1), state=TaskState.executing)
            send_message(chat_id, "✅ Credentials saved. Starting execution...")
            process_task(store.get_active_task())
            return
        elif task.state == TaskState.planning:
            # User is providing feedback for plan revision
            store.update_task(task.task_id, state=TaskState.planning)
            process_task(store.get_active_task())
            return

    # New task
    task_id = f"task-{user_id}-{int(time.time())}"
    task = Task(task_id=task_id, user_id=user_id, chat_id=chat_id, prompt=text)
    store.add_task(task)
    send_message(chat_id, f"📝 New task queued: `{task_id}`\nState: {task.state.value}")
    if store.get_active_task() and store.get_active_task().task_id == task_id:
        process_task(task)
    else:
        send_message(chat_id, "⏳ Task queued. It will start when the current task completes.")


@app.get("/setwebhook")
async def set_webhook():
    service_name = os.getenv("RENDER_SERVICE_NAME", "telegram-coding-agent")
    url = f"https://{service_name}.onrender.com/webhook/{BOT_TOKEN}"
    resp = _tg("setWebhook", {"url": url})
    return JSONResponse(resp)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
