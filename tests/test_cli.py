import subprocess


def test_main():
    assert subprocess.check_output(["cli", "foo", "foobar", "bar"], text=True) == "foobar\n"
