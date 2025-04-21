# File: ai_shell_agent/toolsets/files/integration.py

from pathlib import Path
from rapidfuzz import process
from typing import List, Union

def find_files(
    pattern: str,
    directory: Union[str, Path] = ".",
    glob: str = "**/*",
    fuzzy: bool = True,
    threshold: int = 70,
    limit: int = 50,
    workers: int = 4
) -> List[Path]:
    """
    Search for files under `directory` matching the fixed `glob` pattern.
    If fuzzy=True, filter candidates via RapidFuzz similarity to `pattern`.
    """
    # Resolve start directory for cross‑platform consistency
    start = Path(directory).resolve()

    # Gather everything under `glob`
    candidates = list(start.rglob(glob))
    files = [p for p in candidates if p.is_file()]

    # Relative names for matching
    names = [str(p.relative_to(start)) for p in files]

    if fuzzy:
        matches = process.extract(
            pattern, names,
            limit=limit,
            score_cutoff=threshold
        )
        # matches: List of (matched_name, score, index)
        return [files[idx] for _, _, idx in matches]
    else:
        filtered = [
            files[i] for i, name in enumerate(names)
            if pattern.lower() in name.lower()
        ]
        return filtered[:limit]
