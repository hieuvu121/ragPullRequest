from github import Github

#responsible for calling github api
class GithubClient:
    def __init__(self, token: str):
        self.token = token

    def _gh(self):
        return Github(self.token)

    def get_diff(self, repo_full_name: str, pr_number: int) -> str:
        repo = self._gh().get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        lines: list[str] = []
        for f in pr.get_files():
            if f.patch:
                lines.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch}")
        return "\n".join(lines)

    def post_review(
            self,
            repo_full_name: str,
            pr_number: int,
            summary: str,
            comments: list[dict],
    ) -> int:
        repo = self._gh().get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        gh_comments = [
            {"path": c["path"], "line": c["line"], "side": c.get("side", "RIGHT"), "body": c["body"]}
            for c in comments
        ]
        review = pr.create_review(body=summary, event="COMMENT", comments=gh_comments)
        return review.id

