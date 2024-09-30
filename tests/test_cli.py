import subprocess


def test_main():
    assert subprocess.check_output(["laser-cholera", "foo", "foobar"], text=True) == "foobar\n"
