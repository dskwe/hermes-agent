from hermes_cli.update_cmd_git import _print_fetch_failure


def test_print_fetch_failure_preserves_bounded_git_stderr(capsys):
    stderr = "\n".join(["BUG: pack failure", "fatal: could not finish pack-objects", "fatal: index-pack failed"])

    _print_fetch_failure(stderr)

    output = capsys.readouterr().out
    assert "BUG: pack failure" in output
    assert "fatal: could not finish pack-objects" in output
    assert "fatal: index-pack failed" in output


def test_print_fetch_failure_bounds_long_git_stderr(capsys):
    _print_fetch_failure("\n".join(f"line-{i}" for i in range(20)))

    output = capsys.readouterr().out
    assert "line-0" in output
    assert "line-14" in output
    assert "line-15" not in output
    assert "5 more stderr lines omitted" in output
