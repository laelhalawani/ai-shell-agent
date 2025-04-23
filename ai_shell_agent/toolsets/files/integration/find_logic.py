"""Core logic for finding files, supporting fuzzy matching."""

import os
from pathlib import Path
from typing import List, Union, Tuple, Optional

# Import rapidfuzz safely
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from .... import logger

def find_files_with_logic(
    pattern: str,
    directory: Union[str, Path] = ".",
    glob_pattern: str = "**/*",
    fuzzy: bool = True,
    threshold: int = 70,
    limit: int = 50
) -> Tuple[Optional[List[Path]], Optional[str]]:
    """
    Search for files under `directory` matching `glob_pattern`.
    If fuzzy=True and rapidfuzz is available, filter candidates via similarity.
    Otherwise, use case-insensitive substring matching.

    Returns:
        Tuple[Optional[List[Path]], Optional[str]]: A tuple containing:
            - A list of found Path objects (or None if a critical error occurred like dir not found).
            - An optional error/warning string (e.g., for permission errors).
    """
    if fuzzy and not RAPIDFUZZ_AVAILABLE:
        logger.warning("Rapidfuzz library not found. Falling back to non-fuzzy search.")
        fuzzy = False # Force non-fuzzy if library is missing

    try:
        # Resolve start directory with pathlib for cross-platform consistency
        start_dir = Path(directory).resolve()

        if not start_dir.is_dir():
            logger.error(f"Search directory does not exist or is not a directory: {start_dir}")
            return None, f"Search directory does not exist or is not a directory: {start_dir}"

        logger.debug(f"Starting file search: pattern='{pattern}', directory='{start_dir}', fuzzy={fuzzy}, threshold={threshold}, limit={limit}")

        candidates = []
        permission_warning = None
        try:
            # Use rglob to find all potential candidates matching the glob pattern
            # Note: rglob itself can raise PermissionError on inaccessible subdirs
            # Iterating allows partial results before error
            iterator = start_dir.rglob(glob_pattern)
            while True:
                try:
                     candidate = next(iterator)
                     if candidate.is_file(): # Ensure it's a file
                          candidates.append(candidate)
                except StopIteration:
                     break # End of iteration
                except PermissionError:
                     logger.warning(f"Permission denied accessing subdirectory during glob search under {start_dir}. Search may be incomplete.")
                     permission_warning = f"Search incomplete due to permissions issue under {start_dir}."
                     # Continue searching other accessible directories
                     continue
                except OSError as e:
                     # Catch other potential OS errors during iteration
                     logger.warning(f"OS error during glob search under {start_dir}: {e}. Search may be incomplete.")
                     permission_warning = f"Search incomplete due to OS error under {start_dir}: {e}."
                     continue # Try to continue

        except PermissionError:
             # This catches PermissionError if start_dir itself is inaccessible
             logger.error(f"Permission denied accessing search directory: {start_dir}")
             return None, f"Permission denied accessing search directory: {start_dir}."

        if not candidates:
            logger.debug("No candidate files found matching glob.")
            return [], permission_warning # Return empty list and any permission warning

        # Prepare names relative to start for matching
        # Use Path objects directly where possible
        relative_names = [str(p.relative_to(start_dir)) for p in candidates]
        logger.debug(f"Found {len(candidates)} candidate files.")

        matched_indices = []
        if fuzzy and RAPIDFUZZ_AVAILABLE:
            logger.debug("Performing fuzzy search with RapidFuzz...")
            # Use RapidFuzz to get top-N similar filenames above threshold
            # process.extract returns list of tuples: (match_string, score, index)
            matches = process.extract(
                pattern,
                relative_names,
                scorer=fuzz.WRatio, # Using WRatio often gives good results for filenames
                limit=limit,
                score_cutoff=threshold
            )
            matched_indices = [index for _, _, index in matches]
            logger.debug(f"Fuzzy matches found: {len(matched_indices)}")
        else:
            logger.debug("Performing simple case-insensitive substring search...")
            # Simple case-insensitive substring match
            pattern_lower = pattern.lower()
            temp_indices = [
                i for i, name in enumerate(relative_names)
                if pattern_lower in name.lower()
            ]
            matched_indices = temp_indices[:limit] # Apply limit
            logger.debug(f"Substring matches found: {len(matched_indices)}")

        # Return the actual Path objects corresponding to the matched indices
        final_results = [candidates[idx] for idx in matched_indices]
        return final_results, permission_warning

    except Exception as e:
        logger.error(f"Unexpected error during file search: {e}", exc_info=True)
        return None, f"Unexpected error finding files: {e}"