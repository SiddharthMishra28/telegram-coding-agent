import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class TaskState(str, Enum):
    queued = "queued"
    planning = "planning"
    awaiting_approval = "awaiting_approval"
    awaiting_repo = "awaiting_repo"
    awaiting_pat = "awaiting_pat"
    executing = "executing"
    pushing = "pushing"
    monitoring_actions = "monitoring_actions"
    paused_rate_limit = "paused_rate_limit"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Task:
    def __init__(
        self,
        task_id: str,
        user_id: int,
        chat_id: int,
        prompt: str,
        state: TaskState = TaskState.queued,
    ):
        self.task_id = task_id
        self.user_id = user_id
        self.chat_id = chat_id
        self.prompt = prompt
        self.state = state
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.plan: str | None = None
        self.repo_name: str | None = None
        self.github_username: str | None = None
        self.github_pat: str | None = None
        self.branch: str = "main"
        self.files: list[str] = []
        self.run_url: str | None = None
        self.pages_url: str | None = None
        self.error: str | None = None
        self.result: str | None = None
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.resume_at: str | None = None
        self.checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "prompt": self.prompt,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan": self.plan,
            "repo_name": self.repo_name,
            "github_username": self.github_username,
            "github_pat": self.github_pat,
            "branch": self.branch,
            "files": self.files,
            "run_url": self.run_url,
            "pages_url": self.pages_url,
            "error": self.error,
            "result": self.result,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "resume_at": self.resume_at,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        task = cls(
            task_id=data["task_id"],
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            prompt=data["prompt"],
            state=TaskState(data["state"]),
        )
        task.updated_at = data.get("updated_at", task.created_at)
        task.plan = data.get("plan")
        task.repo_name = data.get("repo_name")
        task.github_username = data.get("github_username")
        task.github_pat = data.get("github_pat")
        task.branch = data.get("branch", "main")
        task.files = data.get("files", [])
        task.run_url = data.get("run_url")
        task.pages_url = data.get("pages_url")
        task.error = data.get("error")
        task.result = data.get("result")
        task.attempts = data.get("attempts", 0)
        task.max_attempts = data.get("max_attempts", 3)
        task.resume_at = data.get("resume_at")
        task.checkpoint_path = data.get("checkpoint_path")
        return task


class TaskStore:
    def __init__(self, path: str = "task_state.json"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._active_task: str | None = None
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._tasks = {k: Task.from_dict(v) for k, v in data.get("tasks", {}).items()}
                self._active_task = data.get("active_task")
            except Exception:
                self._tasks = {}
                self._active_task = None

    def _save(self):
        data = {
            "tasks": {k: v.to_dict() for k, v in self._tasks.items()},
            "active_task": self._active_task,
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)

    def add_task(self, task: Task) -> Task:
        with self._lock:
            self._tasks[task.task_id] = task
            if not self._active_task:
                self._active_task = task.task_id
                task.state = TaskState.planning
            else:
                task.state = TaskState.queued
            self._save()
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_active_task(self) -> Task | None:
        if self._active_task:
            return self._tasks.get(self._active_task)
        return None

    def get_user_tasks(self, user_id: int) -> list[Task]:
        return [t for t in self._tasks.values() if t.user_id == user_id]

    def update_task(self, task_id: str, **updates) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
        return task

    def set_active(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._active_task = task_id
            self._tasks[task_id].state = TaskState.planning
            self._save()
        return True

    def complete_active(self) -> Task | None:
        with self._lock:
            if not self._active_task:
                return None
            task = self._tasks[self._active_task]
            task.state = TaskState.completed
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            next_task = self._find_next_queued()
            if next_task:
                self._active_task = next_task.task_id
                next_task.state = TaskState.planning
            else:
                self._active_task = None
            self._save()
            return task

    def fail_active(self, error: str) -> Task | None:
        with self._lock:
            if not self._active_task:
                return None
            task = self._tasks[self._active_task]
            task.state = TaskState.failed
            task.error = error
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            next_task = self._find_next_queued()
            if next_task:
                self._active_task = next_task.task_id
                next_task.state = TaskState.planning
            else:
                self._active_task = None
            self._save()
            return task

    def cancel_active(self) -> Task | None:
        with self._lock:
            if not self._active_task:
                return None
            task = self._tasks[self._active_task]
            task.state = TaskState.cancelled
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            next_task = self._find_next_queued()
            if next_task:
                self._active_task = next_task.task_id
                next_task.state = TaskState.planning
            else:
                self._active_task = None
            self._save()
            return task

    def requeue_active(self) -> Task | None:
        with self._lock:
            if not self._active_task:
                return None
            task = self._tasks[self._active_task]
            if task.attempts < task.max_attempts:
                task.state = TaskState.queued
                task.attempts += 1
                task.error = None
            else:
                task.state = TaskState.failed
                task.error = "Max retries exceeded"
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            next_task = self._find_next_queued()
            if next_task:
                self._active_task = next_task.task_id
                next_task.state = TaskState.planning
            else:
                self._active_task = None
            self._save()
            return task

    def _find_next_queued(self) -> Task | None:
        for task in self._tasks.values():
            if task.state == TaskState.queued:
                return task
        return None

    def get_paused_tasks(self) -> list[Task]:
        now = datetime.now(timezone.utc).isoformat()
        return [t for t in self._tasks.values() if t.state == TaskState.paused_rate_limit and t.resume_at and t.resume_at <= now]

    def all_tasks(self) -> list[Task]:
        return list(self._tasks.values())
