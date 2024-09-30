from laser_cholera import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
