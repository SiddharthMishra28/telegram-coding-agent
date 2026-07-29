import logging
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

logger = logging.getLogger("keyboards")


def build(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def plan_approval_keyboard(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [build("✅ Approve", f"plan:approve:{task_id}"), build("❌ Reject", f"plan:reject:{task_id}")],
    ])


def task_control_keyboard(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [build("⏸ Pause", f"task:pause:{task_id}"), build("▶ Resume", f"task:resume:{task_id}")],
        [build("⏹ Cancel", f"task:cancel:{task_id}")],
    ])
