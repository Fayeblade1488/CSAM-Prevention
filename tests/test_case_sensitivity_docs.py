import subprocess
from pathlib import Path


def test_docs_has_single_architecture_file():
    docs = Path("docs")
    upper = docs / "ARCHITECTURE.md"
    assert upper.exists(), "docs/ARCHITECTURE.md must exist"

    # Verify git tracking does not include both case-variants
    result = subprocess.run(
        ["git", "ls-files", "docs"], capture_output=True, text=True, check=True
    )
    tracked = [p.strip() for p in result.stdout.splitlines() if p.strip()]
    assert "docs/ARCHITECTURE.md" in tracked
    assert "docs/architecture.md" not in tracked, (
        "Git index still tracks both case-variants; should only track docs/ARCHITECTURE.md"
    )
