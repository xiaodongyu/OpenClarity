import re


def structure_text(tokens: list[dict]) -> str:
    """Sort tokens into reading order and group into lines by vertical proximity."""
    if not tokens:
        return ""

    # Sort by top-left y then x
    sorted_tokens = sorted(
        tokens,
        key=lambda t: (min(pt[1] for pt in t["bbox"]), min(pt[0] for pt in t["bbox"])),
    )

    lines: list[list[dict]] = []
    for token in sorted_tokens:
        top_y = min(pt[1] for pt in token["bbox"])
        bottom_y = max(pt[1] for pt in token["bbox"])
        placed = False
        for line in lines:
            # Check vertical overlap with this line group
            line_top = min(min(pt[1] for pt in t["bbox"]) for t in line)
            line_bottom = max(max(pt[1] for pt in t["bbox"]) for t in line)
            overlap = min(bottom_y, line_bottom) - max(top_y, line_top)
            if overlap > -10:  # within 10 px vertical proximity
                line.append(token)
                placed = True
                break
        if not placed:
            lines.append([token])

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: min(pt[0] for pt in t["bbox"]))
        text_lines.append(" ".join(t["text"] for t in line_sorted))

    return "\n".join(text_lines)


def format_for_speech(text: str) -> str:
    """Prepare structured text for TTS output."""
    if not text:
        return ""

    # Replace newlines with pauses
    result = text.replace("\n", ". ")

    # Announce field boundaries (e.g. "Name: John" → "Name: John")
    result = re.sub(r"(\w+):\s*", r"\1: ", result)

    # Strip repeated whitespace
    result = re.sub(r" {2,}", " ", result).strip()

    # Remove trailing period-space if text ends with ". "
    if result.endswith(". "):
        result = result[:-2]

    return result
