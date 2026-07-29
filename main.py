import logging
import os
import re
import sys
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from agent import create_agent
from personality import PersonalityConfig
from langchain_core.messages import HumanMessage

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

personality = PersonalityConfig.from_env()
conversations: dict[int, list] = {}


def send_message(chat_id: int, text: str):
    logger.info("Sending message to chat %s: %s", chat_id, text[:200])
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


def extract_github_token(text: str) -> tuple[str, str]:
    match = re.search(r"github\s*(?:pat|token|personal\s*access\s*token)?\s*[:\-]?\s*([A-Za-z0-9_]{30,})", text, re.IGNORECASE)
    username_match = re.search(r"github\s*(?:username)?\s*[:\-]?\s*([A-Za-z0-9](?:[A-Za-z0-9\-]{1,38}))", text, re.IGNORECASE)
    token = match.group(1) if match else os.getenv("GITHUB_PAT", "")
    username = username_match.group(1) if username_match else os.getenv("GITHUB_USERNAME", "")
    return token, username


@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        logger.warning("Unauthorized webhook attempt with token %s", token)
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    update = await request.json()
    message = update.get("message")
    if not message or "text" not in message:
        return JSONResponse({"ok": True})

    user_id = message["from"]["id"]
    chat_id = message["chat"]["id"]
    text = message["text"].strip()
    logger.info("Received message from user %s chat %s: %s", user_id, chat_id, text[:200])

    if text == "/start":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_welcome_message())
        return JSONResponse({"ok": True})

    if text == "/reset":
        conversations.pop(user_id, None)
        send_message(chat_id, personality.get_reset_message())
        return JSONResponse({"ok": True})

    if text == "/help":
        send_message(chat_id, personality.get_help_message())
        return JSONResponse({"ok": True})

    if text.startswith("/personality"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            send_message(chat_id, "Usage: `/personality <friendly|professional|concise|verbose|custom>`")
            return JSONResponse({"ok": True})
        tone = parts[1].strip().lower()
        valid = {"friendly", "professional", "concise", "verbose", "custom"}
        if tone not in valid:
            send_message(chat_id, f"Unknown tone. Choose from: {', '.join(sorted(valid))}")
            return JSONResponse({"ok": True})
        if tone == "custom":
            send_message(chat_id, "Set `AGENT_CUSTOM_PERSONA` in your Render env vars to a custom system prompt.")
            return JSONResponse({"ok": True})
        personality.tone = tone
        send_message(chat_id, f"Personality updated to *{tone}*. From now on I'll behave accordingly.")
        return JSONResponse({"ok": True})

    send_typing(chat_id)
    workspace_dir = os.path.join(os.getcwd(), "workspace", str(user_id))
    os.makedirs(workspace_dir, exist_ok=True)

    try:
        gh_token, gh_username = extract_github_token(text)
        logger.info("Extracted GitHub username=%s token_present=%s", bool(gh_username), bool(gh_token))

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
        logger.exception("Failed to process message")
        send_message(chat_id, f"⚠️ Something went wrong: `{e}`")

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
