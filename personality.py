import os
from pydantic import BaseModel


class PersonalityConfig(BaseModel):
    name: str = "CodeAgent"
    tone: str = "friendly"
    custom_persona: str | None = None
    emoji: bool = True

    @classmethod
    def from_env(cls) -> "PersonalityConfig":
        return cls(
            name=os.getenv("AGENT_NAME", "CodeAgent"),
            tone=os.getenv("AGENT_TONE", "friendly"),
            custom_persona=os.getenv("AGENT_CUSTOM_PERSONA"),
            emoji=os.getenv("AGENT_EMOJI", "true").lower() == "true",
        )

    def get_system_prompt(self) -> str:
        if self.custom_persona:
            return self.custom_persona.strip()

        base = f"You are {self.name}, an expert coding assistant."

        tone_behavior = {
            "friendly": (
                " Be warm, encouraging, and approachable. "
                "Celebrate successes and normalize mistakes. "
                "Use simple language and add light emojis occasionally."
            ),
            "professional": (
                " Be precise, concise, and formal. "
                "Focus on correctness, best practices, and clean architecture. "
                "Avoid filler words and unnecessary commentary."
            ),
            "concise": (
                " Be extremely brief. "
                "Return only the essential answer or code. "
                "Skip explanations unless explicitly asked."
            ),
            "verbose": (
                " Be thorough and educational. "
                "Explain your reasoning, tradeoffs, and alternatives. "
                "Teach concepts as you implement them."
            ),
        }

        rules = [
            "Write clean, well-documented code.",
            "Always read files before editing them.",
            "Run code after writing it to verify it works.",
            "If something fails, inspect the error, fix it, and retry.",
            "Keep functions small and focused.",
            "Use type hints in Python.",
        ]

        behavior = tone_behavior.get(self.tone, tone_behavior["friendly"])
        return base + behavior + "\n\n" + "\n".join(f"- {r}" for r in rules)

    def get_welcome_message(self) -> str:
        e = "✨" if self.emoji else ""
        tone_label = self.tone.capitalize()
        return (
            f"{e} Hi! I'm *{self.name}* — your personal coding assistant.\n\n"
            f"Current mode: *{tone_label}*\n"
            "Send me any coding task and I will build it step by step.\n\n"
            "Commands:\n"
            "`/start` — Show this welcome\n"
            "`/reset` — Clear our conversation\n"
            "`/personality` `<friendly|professional|concise|verbose|custom>` — Switch behavior\n"
            "`/help` — Show commands\n\n"
            "What would you like to build?"
        )

    def get_help_message(self) -> str:
        e = "🛠️" if self.emoji else ""
        return (
            f"{e} *{self.name}* can help you write, read, and run code.\n\n"
            "Just describe what you want in plain English. Examples:\n"
            "• \"Write a Python script that prints the first 10 Fibonacci numbers\"\n"
            "• \"Create a calculator.py with add and multiply functions\"\n"
            "• \"Read main.py and suggest improvements\"\n\n"
            "Use `/personality` to change how I behave."
        )

    def get_reset_message(self) -> str:
        e = "🔄" if self.emoji else ""
        return f"{e} Conversation cleared! Starting fresh."
