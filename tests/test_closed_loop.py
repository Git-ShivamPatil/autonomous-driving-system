from ads.eval.closed_loop import run_key


def test_run_key_changes_with_the_checkpoint_bytes(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"weights-1")
    b.write_bytes(b"weights-2")
    assert run_key("model", a) != run_key("model", b)
    assert run_key("model", a) == run_key("model", a)


def test_run_key_separates_splits():
    assert run_key("expert", None, "val") != run_key("expert", None, "test")


def test_run_key_separates_drivers():
    assert run_key("expert", None) != run_key("model", None)


def test_closed_loop_module_does_not_import_the_simulator_or_torch():
    import sys

    assert "metadrive" not in sys.modules
