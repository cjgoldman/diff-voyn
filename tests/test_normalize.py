"""Task 0.3 acceptance: shared normalizer — uniform, whitespace-free, accounted."""

from diff_voyn.normalize import NormStats, normalize
from diff_voyn.vocab import LETTERS


def test_lowercase_and_alphabet_closure():
    out = normalize("Gallia est OMNIS divisa")
    assert out == "galliaestomnisdivisa"
    assert set(out) <= set(LETTERS)


def test_whitespace_fully_removed_and_counted():
    stats = NormStats()
    out = normalize("a b\tc\nd e f", stats)
    assert out == "abcdef"
    assert stats.whitespace_removed == 5
    assert not any(c.isspace() for c in out)


def test_j_merges_to_i_but_u_v_stay_distinct():
    assert normalize("Jam iuvat Justitia") == "iamiuvatiustitia"
    assert normalize("Jj") == "ii"


def test_k_w_preserved_not_lossy_mapped():
    # design §2: extension, not Greshko's k->c / w->uu
    assert normalize("Kraft Wasser") == "kraftwasser"


def test_diacritics_and_ligatures_fold_uniformly():
    assert normalize("überflüßig") == "uberflussig"  # ü→u (not ue), ß→ss
    assert normalize("cæsar œuvre") == "caesaroeuvre"
    assert normalize("señor àéîõû") == "senoraeiou"


def test_long_s_folds():
    assert normalize("Waſſer") == "wasser"


def test_punct_digits_counted_separately_from_letter_drops():
    stats = NormStats()
    out = normalize("anno 1404, cap. VII: αβγ", stats)
    assert out == "annocapvii"
    assert stats.foreign_script_removed == 3  # the Greek letters
    assert stats.letters_dropped == 0  # no Latin-script loss
    assert stats.punct_digit_removed == 7  # 1404 , . :
    assert set(stats.foreign_examples) == {"α", "β", "γ"}


def test_medieval_abbreviature_letters_mapped_not_dropped():
    stats = NormStats()
    # r rotunda, ezh, q-with-stroke, v-with-stroke — attested in DTA sources
    assert normalize("deꝛ herꝛ ʒu ꝙuod ꝟon", stats) == "derherrzuquodvon"
    assert stats.letters_dropped == 0


def test_unmapped_latin_letter_still_counts_as_drop():
    stats = NormStats()
    normalize("aƕb", stats)  # hwair has no mapping — genuine loss
    assert stats.letters_dropped == 1
    assert stats.foreign_script_removed == 0


def test_idempotent():
    for s in ["Vermögen, & Kraft!", "Jam 123 œß", "ipsa scientia potestas est"]:
        once = normalize(s)
        assert normalize(once) == once


def test_rate_properties():
    stats = NormStats()
    normalize("abc αβγ ƕ", stats)
    assert stats.alphabetic_input == 4  # abc + hwair
    assert abs(stats.letter_drop_rate - 0.25) < 1e-9
    assert abs(stats.foreign_script_rate - 3 / 7) < 1e-9
