import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("planner")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )


PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior full-stack architect. Create a detailed implementation plan for the user's request.

Output ONLY a numbered plan in this exact format:
1. Overview: 1-2 sentence summary
2. Tech stack: list of technologies
3. File structure: each file with its purpose
4. Implementation steps: numbered list of all steps needed
5. Testing/verification: how to verify it works
6. Deployment: deployment notes

Keep it concise but complete. Do NOT write code in the plan. Do NOT add extra commentary."""),
    ("user", "{prompt}"),
])


REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior full-stack architect. The user rejected your previous plan. Create a revised plan incorporating their feedback.

Output ONLY a numbered plan in the same format as before:
1. Overview
2. Tech stack
3. File structure
4. Implementation steps
5. Testing/verification
6. Deployment

Incorporate the user's feedback into the revised plan."""),
    ("user", "Original request: {prompt}\n\nFeedback: {feedback}"),
])


def generate_plan(prompt: str) -> str:
    llm = get_llm()
    chain = PLAN_PROMPT | llm | StrOutputParser()
    plan = chain.invoke({"prompt": prompt})
    logger.info("Generated plan: %s", plan[:200])
    return plan


def revise_plan(prompt: str, feedback: str) -> str:
    llm = get_llm()
    chain = REVISION_PROMPT | llm | StrOutputParser()
    plan = chain.invoke({"prompt": prompt, "feedback": feedback})
    logger.info("Revised plan: %s", plan[:200])
    return plan
