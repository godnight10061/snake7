def test_import_eval_module():
    import snake7.eval as eval_mod

    assert hasattr(eval_mod, "main")

