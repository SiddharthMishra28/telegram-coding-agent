import base64
import json
import logging
import os
import time
import urllib.request
from typing import Any

logger = logging.getLogger("github")


class GitHubManager:
    def __init__(self, token: str, username: str):
        self.token = token
        self.username = username
        self.api = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        url = f"{self.api}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode()
            logger.error("GitHub API error %s %s: %s %s", method, path, e.code, body_text)
            raise RuntimeError(f"GitHub API {e.code}: {body_text}") from e

    def create_repo(self, name: str) -> dict:
        logger.info("Creating repo %s/%s", self.username, name)
        return self._request("PUT", f"/repos/{self.username}/{name}", {"name": name, "private": False})

    def ensure_pages(self, repo_name: str, branch: str = "main"):
        logger.info("Enabling Pages for %s/%s", self.username, repo_name)
        try:
            return self._request("POST", f"/repos/{self.username}/{repo_name}/pages", {"source": {"branch": branch, "path": "/"}})
        except RuntimeError as e:
            if "already" in str(e).lower() or "409" in str(e):
                return {"status": "already_enabled"}
            raise

    def create_file(self, repo_name: str, path: str, content: str, message: str, branch: str = "main") -> dict:
        logger.info("Creating file %s in %s/%s", path, self.username, repo_name)
        return self._request(
            "PUT",
            f"/repos/{self.username}/{repo_name}/contents/{path}",
            {
                "message": message,
                "content": base64.b64encode(content.encode()).decode(),
                "branch": branch,
            },
        )

    def update_file(self, repo_name: str, path: str, content: str, message: str, sha: str, branch: str = "main") -> dict:
        logger.info("Updating file %s in %s/%s", path, self.username, repo_name)
        return self._request(
            "PUT",
            f"/repos/{self.username}/{repo_name}/contents/{path}",
            {
                "message": message,
                "content": base64.b64encode(content.encode()).decode(),
                "sha": sha,
                "branch": branch,
            },
        )

    def get_file(self, repo_name: str, path: str, branch: str = "main") -> dict | None:
        try:
            return self._request("GET", f"/repos/{self.username}/{repo_name}/contents/{path}?ref={branch}")
        except RuntimeError:
            return None

    def list_files(self, repo_name: str, branch: str = "main") -> list[str]:
        try:
            data = self._request("GET", f"/repos/{self.username}/{repo_name}/git/trees/{branch}?recursive=1")
            return [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]
        except RuntimeError:
            return []

    def create_workflow(self, repo_name: str, workflow_content: str):
        path = f".github/workflows/deploy.yml"
        return self.create_file(repo_name, path, workflow_content, "Add GitHub Pages deployment workflow")

    def trigger_workflow(self, repo_name: str, workflow_file: str = "deploy.yml", branch: str = "main") -> dict:
        logger.info("Triggering workflow %s on %s/%s", workflow_file, self.username, repo_name)
        return self._request("POST", f"/repos/{self.username}/{repo_name}/actions/workflows/{workflow_file}/dispatches", {"ref": branch})

    def get_workflow_runs(self, repo_name: str, workflow_file: str = "deploy.yml", per_page: int = 5) -> list[dict]:
        data = self._request("GET", f"/repos/{self.username}/{repo_name}/actions/workflows/{workflow_file}/runs?per_page={per_page}")
        return data.get("workflow_runs", [])

    def get_run_logs(self, repo_name: str, run_id: int) -> str:
        url = f"{self.api}/repos/{self.username}/{repo_name}/actions/runs/{run_id}/logs"
        req = urllib.request.Request(url, headers=self.headers, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                logs_data = json.loads(resp.read().decode())
                return logs_data.get("logs", "")
        except Exception as e:
            logger.error("Failed to fetch logs for run %s: %s", run_id, e)
            return f"Failed to fetch logs: {e}"

    def wait_for_run(self, repo_name: str, workflow_file: str = "deploy.yml", timeout: int = 300, poll_interval: int = 10) -> dict:
        deadline = time.time() + timeout
        last_run_id = None
        while time.time() < deadline:
            runs = self.get_workflow_runs(repo_name, workflow_file)
            if runs:
                run = runs[0]
                last_run_id = run["id"]
                status = run.get("status")
                conclusion = run.get("conclusion")
                logger.info("Workflow run %s status=%s conclusion=%s", run["id"], status, conclusion)
                if status == "completed":
                    logs = self.get_run_logs(repo_name, run["id"])
                    return {
                        "id": run["id"],
                        "status": status,
                        "conclusion": conclusion,
                        "html_url": run.get("html_url"),
                        "logs": logs[-4000:] if logs else "",
                    }
            time.sleep(poll_interval)
        raise TimeoutError(f"Workflow did not complete within {timeout} seconds")
