import os

def pytest_sessionstart(session):
    # keep unit tests fast & offline
    os.environ.setdefault("DISABLE_NLP", "1")
