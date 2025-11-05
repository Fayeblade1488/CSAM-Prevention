import os
import platform
from pathlib import Path


def test_docs_has_single_architecture_file():
    docs = Path('docs')
    upper = docs / 'ARCHITECTURE.md'
    lower = docs / 'architecture.md'
    # On case-insensitive filesystems, both names collide; repo should not track both.
    assert upper.exists(), "docs/ARCHITECTURE.md must exist"
    assert not (upper.exists() and lower.exists() and os.path.samefile(upper, lower)), (
        "Repository should not track both docs/ARCHITECTURE.md and docs/architecture.md;"
        " this breaks case-insensitive filesystems"
    )