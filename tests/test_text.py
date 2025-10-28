from csam_guard.guard import CSAMGuard, DEFAULT_CONFIG

def test_text_core_paths():
    g = CSAMGuard(DEFAULT_CONFIG)
    assert not g.assess("A 15-year-old in school uniform").allow
    assert g.assess("18+ adult woman").allow
    # spelling
    d = g.assess("twenty five year old")
    assert 25 in d.signals["ages"]
    assert d.allow
    d = g.assess("one hundred year old")
    assert 100 in d.signals["ages"]
    assert d.allow
