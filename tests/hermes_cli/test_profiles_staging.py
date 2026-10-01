from hermes_cli import profiles


def test_clone_staging_dir_is_unique_per_call(tmp_path):
    profile_dir = tmp_path / "profiles" / "demo"

    first = profiles._clone_staging_dir(profile_dir)
    second = profiles._clone_staging_dir(profile_dir)

    assert first != second
    assert first.parent == second.parent == profile_dir.parent