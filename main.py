import os
import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from agent import create_agent
from personality import PersonalityConfig
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

personality = PersonalityConfig.from_env()

conversations: dict[int, list] = {}


def send_message(chat_id: int, text: str):
    with httpx.Client() as client:
        client.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })


def send_typing(chat_id: int):
    with httpx.Client() as client:
        client.post(f"{BASE_URL}/sendChatAction", json={
            "chat_id": chat_id,
            "action": "typing",
        })


def process_message(user_id: int, chat_id: int, text: str):
    text = text.strip()

    if text == "/start":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_welcome_message())
        return

    if text == "/reset":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_reset_message())
        return

    if text == "/help":
        send_message(chat_id, personality.get_help_message())
        return

    if text.startswith("/personality"):
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
            send_message(chat_id, "Set `AGENT_CUSTOM_PERSONA` in your Render env vars to a custom system prompt.")
            return
        personality.tone = tone
        send_message(chat_id, f"Personality updated to *{tone}*. From now on I'll behave accordingly.")
        return

    send_typing(chat_id)
    workspace_dir = os.path.join(os.getcwd(), "workspace", str(user_id))
    os.makedirs(workspace_dir, exist_ok=True)

    try:
        agent = create_agent(personality.get_system_prompt(), workspace_dir)
        history = conversations.get(user_id, [])
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=text)]},
            config={"recursion_limit": 50},
        )
        response = result["messages"][-1].content
        conversations[user_id] = result["messages"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n...(truncated)"
        send_message(chat_id, response)
    except Exception as e:
        send_message(chat_id, f"⚠️ Something went wrong: `{e}`")


@app.post("/webhook/{token}")
async def webhook(token: str, request: Request, background_tasks: BackgroundTasks):
    if token != BOT_TOKEN:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    update = await request.json()
    message = update.get("message")
    if not message or "text" not in message:
        return JSONResponse({"ok": True})

    user_id = message["from"]["id"]
    chat_id = message["chat"]["id"]
    text = message["text"]

    background_tasks.add_task(process_message, user_id, chat_id, text)
    return JSONResponse({"ok": True})


@app.get("/setwebhook")
async def set_webhook():
    service_name = os.getenv("RENDER_SERVICE_NAME", "telegram-coding-agent")
    url = f"https://{service_name}.onrender.com/webhook/{BOT_TOKEN}"
    with httpx.Client() as client:
        resp = client.post(f"{BASE_URL}/setWebhook", json={"url": url})
    return JSONResponse(resp.json())


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
