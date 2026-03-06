from __future__ import annotations


def write_output(transcript: str, output_file: str | None) -> None:
    print(transcript)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")
