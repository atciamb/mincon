"""On GitHub Actions every failed test is also written as a workflow annotation.

A job's annotations can be read from the public API
(``/repos/<owner>/<repo>/check-runs/<job id>/annotations``); its log needs a token. Without this a
red Python job says "Process completed with exit code 1" and nothing else. Locally the hook does
nothing.
"""
import os


def _escape(text, in_property=False):
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if in_property:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def pytest_terminal_summary(terminalreporter):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for kind in ("failed", "error"):
        for report in terminalreporter.stats.get(kind, []):
            lines = [line for line in str(report.longrepr).splitlines() if line.strip()]
            body = "\n".join(lines[-30:])[-3500:]
            title = f"{kind}: {getattr(report, 'nodeid', '?')}"
            terminalreporter.write_line(f"::error title={_escape(title, True)}::{_escape(body)}")
