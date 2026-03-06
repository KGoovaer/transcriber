import os
import pytest
from transcriber.output import write_output


def test_prints_to_stdout(capsys):
    write_output("hello world", output_file=None)
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


def test_writes_to_file(tmp_path):
    out = tmp_path / "transcript.txt"
    write_output("hello world", output_file=str(out))
    assert out.read_text().strip() == "hello world"


def test_writes_to_file_and_prints(tmp_path, capsys):
    out = tmp_path / "transcript.txt"
    write_output("hello world", output_file=str(out))
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
    assert out.read_text().strip() == "hello world"
