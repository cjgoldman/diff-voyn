"""Negative-control battery — task 6.3 (design §9.2).

Every control is an *instance* in the presentation format of
``presentations.py`` / ``apply.py`` so it runs through exactly the pipeline
the manuscript runs through (same heads, same outer tier, same MDL scale,
same abstention rule), with its truth attached for the report:

``voynichesque``  (must abstain)
    Output of the pinned ``voynichesque.py`` (structured gibberish with no
    recoverable plaintext) on held-out windows of each inventory language;
    presented as the EVA-like character stream (rungs 1–2) and as word
    tokens (rung 3, Naibbe-parseable subset with coverage).

``shuffled``  (must abstain)
    A held-out window with its letters randomly permuted — the unigram
    statistics of a real language with no sequential structure — under a
    random 1:1 key (the symbol stream then carries no structure at all for
    rungs 1–2 to find).

``contamination``  (confusions documented)
    Out-of-inventory languages — Dutch, English (Germanic), French, Spanish
    (Romance) — from the pinned voynich-attack corpora, normalized to the
    frozen alphabet, enciphered under the in-inventory-fit ciphers: the 1:1
    substitution (rungs 1–2) and Naibbe (rung 3). The question is whether an
    untrained language masquerades as an in-inventory fit, and whether the
    family is at least right.

``positive``  (must NOT abstain)
    The same two encipherments of held-out in-inventory windows — the rule's
    false-abstention rate on true decipherments, measured on the same
    pipeline and lengths as the controls.
"""

from __future__ import annotations

import csv
import json
import re
import zlib
from pathlib import Path

import numpy as np

from ..ciphers.external import data_root
from ..corpus.splits import load_splits
from ..data.loader import LANG_TO_INDEX
from ..heads.ngram import LETTER_TO_IDX, A
from ..normalize import normalize
from ..vocab import LETTERS

LANGS = tuple(LANG_TO_INDEX)
OUT_OF_INVENTORY = {
    "dutch": ("germanic", "dutch/DBNL"),
    "english": ("germanic", "english/EEBO"),
    "french": ("romance", "french"),
    "spanish": ("romance", "spanish"),
}


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(map(str, key)).encode()))


def _letters_to_text(ids: np.ndarray) -> str:
    return "".join(LETTERS[i] for i in ids)


def _text_to_ids(text: str) -> np.ndarray:
    return np.array(
        [LETTER_TO_IDX[c] for c in text if c in LETTER_TO_IDX], dtype=np.int64
    )


# -- out-of-inventory text ---------------------------------------------------


def _textstrings(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        col = (
            "textstring_simple"
            if "textstring_simple" in (r.fieldnames or [])
            else "textstring"
        )
        return [row[col] for row in r if row.get(col)]


def out_of_inventory_docs(
    language: str, root: Path | None = None
) -> list[tuple[str, str]]:
    """[(doc_id, normalized letter stream)] for one out-of-inventory
    language from the pinned voynich-attack corpora."""
    root = root or data_root()
    base = (
        root / "external" / "voynich-attack" / "corpora" / OUT_OF_INVENTORY[language][1]
    )
    docs = []
    for p in sorted(base.rglob("*.csv")):
        if "parsed" in p.name and language != "spanish":
            continue
        text = normalize(" ".join(_textstrings(p)))
        if len(text) >= 5000:
            docs.append((p.stem, text))
    return docs


def sample_out_of_inventory(
    language: str, length: int, rng, root=None
) -> tuple[str, np.ndarray]:
    docs = out_of_inventory_docs(language, root)
    w = np.array([len(t) for _, t in docs], float)
    doc_id, text = docs[rng.choice(len(docs), p=w / w.sum())]
    start = rng.integers(0, len(text) - length + 1)
    return doc_id, _text_to_ids(text[start : start + length])


# -- instance builders -------------------------------------------------------


def _sym_instance(
    name: str, kind: str, symbols: np.ndarray, alphabet: list[str], truth: dict
) -> dict:
    return {
        "name": name,
        "kind": kind,
        "n_symbols": len(alphabet),
        "alphabet": alphabet,
        "n_stream": len(symbols),
        "coverage": {"n_chars": len(symbols), "covered_fraction": 1.0},
        "symbols": symbols.tolist(),
        "truth": truth,
    }


def _words_instance(name: str, tokens: list[str], parser, truth: dict) -> dict:
    ok = [
        t
        for t in tokens
        if (parser.parse_token(t).uni is not None or parser.parse_token(t).bi)
    ]
    n_chars = sum(map(len, tokens))
    n_ok = sum(map(len, ok))
    return {
        "name": name,
        "kind": "words",
        "n_symbols": 0,
        "alphabet": [],
        "n_stream": len(ok),
        "coverage": {
            "n_words": len(tokens),
            "n_parseable_words": len(ok),
            "word_fraction": len(ok) / max(len(tokens), 1),
            "n_chars": n_chars,
            "n_parseable_chars": n_ok,
            "covered_fraction": n_ok / max(n_chars, 1),
        },
        "tokens": ok,
        "truth": truth,
    }


def substitution_instance(name: str, plain: np.ndarray, rng, truth: dict) -> dict:
    perm = rng.permutation(A)
    sym = perm[plain]
    t = dict(
        truth,
        kind="sub1to1",
        plain_ids=plain.tolist(),
        sym_to_letter=np.argsort(perm).tolist(),
    )
    return _sym_instance(name, "eva", sym, [f"s{i}" for i in range(A)], t)


def naibbe_instance(
    name: str, plain: np.ndarray, seed: int, parser, truth: dict
) -> dict:
    from ..ciphers.naibbe import NaibbeCipher

    tokens, segments = NaibbeCipher(seed=seed).encipher(_letters_to_text(plain))
    plain23 = [LETTER_TO_IDX[c] for c in "".join(segments)]
    t = dict(truth, kind="naibbe", plain_ids=plain23, cipher_seed=seed)
    return _words_instance(name, tokens, parser, t)


def voynichesque_instances(
    name: str, text: str, seed: int, parser, truth: dict
) -> list[dict]:
    from ..ciphers.controls import Voynichesque

    gen = Voynichesque()
    out = None
    for attempt in range(40):  # ~20% of parameter draws are infeasible upstream
        try:
            out = gen.generate(text, seed=seed + attempt)
            break
        except ValueError:
            continue
    if out is None:
        raise RuntimeError("voynichesque: no feasible parameter draw")
    tokens = [normalize(w) for w in out.split()]
    tokens = [w for w in tokens if w]
    stream = "".join(tokens)
    alphabet = sorted(set(stream))
    idx = {c: i for i, c in enumerate(alphabet)}
    sym = np.array([idx[c] for c in stream], dtype=np.int64)
    t = dict(truth, kind="voynichesque", n_tokens=len(tokens))
    return [
        _sym_instance(name, "eva", sym, alphabet, t),
        _words_instance(name, tokens, parser, t),
    ]


def build_controls(
    out_dir: Path,
    *,
    per_language: int = 3,
    length: int = 2000,
    naibbe_length: int = 1000,
    seed: int = 0,
    root: Path | None = None,
) -> list[dict]:
    """Write every control instance to ``out_dir`` and return the manifest
    [{name, kind, file, control, truth}]."""
    from ..heads.naibbe_parse import NaibbeParser
    from ..heads.synth import HeldoutSampler

    root = root or data_root()
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    parser = NaibbeParser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(control: str, inst: dict):
        fname = re.sub(r"[^A-Za-z0-9_]+", "_", inst["name"]) + f"_{inst['kind']}.json"
        inst["control"] = control
        (out_dir / fname).write_text(json.dumps(inst))
        manifest.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "file": fname,
                "control": control,
                "truth": {
                    k: v
                    for k, v in inst["truth"].items()
                    if k not in ("plain_ids", "sym_to_letter")
                },
                "coverage": inst["coverage"],
                "n_symbols": inst["n_symbols"],
                "n_stream": inst["n_stream"],
            }
        )

    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        fam = {"latin": "romance", "italian": "romance", "german": "germanic"}[lang]
        for t in range(per_language):
            truth = {"language": lang, "family": fam, "in_inventory": True}
            # positive controls
            rng = _rng("positive", seed, lang, t)
            plain = sampler.sample(length, rng)
            emit(
                "positive",
                substitution_instance(f"positive/{lang}/t{t}", plain, rng, truth),
            )
            rng = _rng("positive-naibbe", seed, lang, t)
            plain_n = sampler.sample(naibbe_length, rng)
            emit(
                "positive",
                naibbe_instance(
                    f"positive/{lang}/t{t}",
                    plain_n,
                    int(rng.integers(2**31)),
                    parser,
                    truth,
                ),
            )
            # shuffled
            rng = _rng("shuffled", seed, lang, t)
            plain = sampler.sample(length, rng)
            shuf = rng.permutation(plain)
            emit(
                "shuffled",
                substitution_instance(
                    f"shuffled/{lang}/t{t}",
                    shuf,
                    rng,
                    dict(truth, source_language=lang),
                ),
            )
            # voynichesque (source content destroyed by construction)
            rng = _rng("voynichesque", seed, lang, t)
            src = sampler.sample(int(length * 0.75), rng)
            for inst in voynichesque_instances(
                f"voynichesque/{lang}/t{t}",
                _letters_to_text(src),
                int(rng.integers(2**31)),
                parser,
                dict(truth, source_language=lang),
            ):
                emit("voynichesque", inst)
    for lang, (fam, _) in OUT_OF_INVENTORY.items():
        for t in range(per_language):
            truth = {"language": lang, "family": fam, "in_inventory": False}
            rng = _rng("contamination", seed, lang, t)
            doc, plain = sample_out_of_inventory(lang, length, rng, root)
            emit(
                "contamination",
                substitution_instance(
                    f"contamination/{lang}/t{t}", plain, rng, dict(truth, doc=doc)
                ),
            )
            rng = _rng("contamination-naibbe", seed, lang, t)
            doc, plain_n = sample_out_of_inventory(lang, naibbe_length, rng, root)
            emit(
                "contamination",
                naibbe_instance(
                    f"contamination/{lang}/t{t}",
                    plain_n,
                    int(rng.integers(2**31)),
                    parser,
                    dict(truth, doc=doc),
                ),
            )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# -- word-level homophonic battery (heads/wordhom.py) ------------------------


def wordhom_instance(
    name: str,
    plain: np.ndarray,
    targets,
    rng,
    truth: dict,
    *,
    n_types: int = 2500,
    n_top: int | None = None,
    zipf_s: float = 1.0,
) -> dict:
    """Held-out plaintext under the synthetic word-homophonic cipher
    (letters + top doubled letters, Zipf homophone usage, repeat rule), presented
    as ``wordtypes<K>`` (top-``n_top`` types, all when ``None``)."""
    from ..heads.wordhom import WordHomCipher
    from .presentations import wordtypes_presentation

    units, toks, sym_to_unit = WordHomCipher(targets, n_types, zipf_s).encipher(
        plain, rng
    )
    words = [f"w{t}" for t in toks]
    pres = wordtypes_presentation("ctrl", "-", n_top, words=words, name=name)
    rec = {
        "name": name,
        "kind": pres.kind,
        "n_symbols": pres.n_symbols,
        "alphabet": pres.alphabet,
        "n_stream": len(pres.symbols),
        "coverage": pres.coverage,
        "symbols": pres.symbols.tolist(),
        "token_pos": pres.token_starts.tolist(),
        "all_tokens": words,
        "truth": dict(
            truth,
            kind="wordhom",
            plain_ids=plain.tolist(),
            unit_ids=units.tolist(),
            bigrams=targets.as_list(),
            # truth unit of every presented symbol id
            sym_to_unit=[int(sym_to_unit[int(w[1:])]) for w in pres.alphabet],
            n_types=len(sym_to_unit),
        ),
    }
    return rec


def voynichesque_wordtypes_instance(name, text, seed, truth, n_top=None) -> dict:
    from ..ciphers.controls import Voynichesque
    from .presentations import wordtypes_presentation

    gen = Voynichesque()
    out = None
    for attempt in range(40):
        try:
            out = gen.generate(text, seed=seed + attempt)
            break
        except ValueError:
            continue
    if out is None:
        raise RuntimeError("voynichesque: no feasible parameter draw")
    words = [w for w in (normalize(w) for w in out.split()) if w]
    pres = wordtypes_presentation("ctrl", "-", n_top, words=words, name=name)
    return {
        "name": name,
        "kind": pres.kind,
        "n_symbols": pres.n_symbols,
        "alphabet": pres.alphabet,
        "n_stream": len(pres.symbols),
        "coverage": pres.coverage,
        "symbols": pres.symbols.tolist(),
        "token_pos": pres.token_starts.tolist(),
        "all_tokens": words,
        "truth": dict(truth, kind="voynichesque", n_tokens=len(words)),
    }


def build_wordhom_controls(
    out_dir: Path,
    evaluator,
    *,
    per_language: int = 3,
    length: int = 8000,
    n_types: int = 2500,
    n_top: int | None = None,
    seed: int = 0,
    root: Path | None = None,
    shapes: list[tuple[str, int, int]] | None = None,
) -> list[dict]:
    """positive / shuffled / voynichesque / contamination instances for the
    word-homophonic head; ``shapes`` = [(tag, length, n_types)] adds one
    positive per language and shape (named ``positive/<lang>/<tag>``) so the
    pipeline's behaviour on a TRUE cipher of the manuscript's own
    type/token shape is measured like-for-like; ``evaluator`` supplies each in-inventory
    language's top doubled letters (out-of-inventory languages use the sampled
    document's own)."""
    from ..heads.synth import HeldoutSampler
    from ..heads.wordhom import language_targets, targets_from_ids

    root = root or data_root()
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(control: str, inst: dict):
        fname = re.sub(r"[^A-Za-z0-9_]+", "_", inst["name"]) + f"_{inst['kind']}.json"
        inst["control"] = control
        (out_dir / fname).write_text(json.dumps(inst))
        manifest.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "file": fname,
                "control": control,
                "truth": {
                    k: v
                    for k, v in inst["truth"].items()
                    if k not in ("plain_ids", "unit_ids", "sym_to_unit")
                },
                "coverage": inst["coverage"],
                "n_symbols": inst["n_symbols"],
                "n_stream": inst["n_stream"],
            }
        )

    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        fam = {"latin": "romance", "italian": "romance", "german": "germanic"}[lang]
        targets = language_targets(evaluator, lang)
        for t in range(per_language):
            truth = {"language": lang, "family": fam, "in_inventory": True}
            rng = _rng("wordhom-positive", seed, lang, t)
            plain = sampler.sample(length, rng)
            emit(
                "positive",
                wordhom_instance(
                    f"positive/{lang}/t{t}",
                    plain,
                    targets,
                    rng,
                    truth,
                    n_types=n_types,
                    n_top=n_top,
                ),
            )
            for tag, ln, nt in shapes or []:
                rng = _rng("wordhom-positive", seed, lang, tag)
                plain = sampler.sample(ln, rng)
                emit(
                    "positive",
                    wordhom_instance(
                        f"positive/{lang}/{tag}",
                        plain,
                        targets,
                        rng,
                        dict(truth, shape=tag),
                        n_types=nt,
                        n_top=n_top,
                    ),
                )
            rng = _rng("wordhom-shuffled", seed, lang, t)
            plain = rng.permutation(sampler.sample(length, rng))
            emit(
                "shuffled",
                wordhom_instance(
                    f"shuffled/{lang}/t{t}",
                    plain,
                    targets,
                    rng,
                    dict(truth, source_language=lang),
                    n_types=n_types,
                    n_top=n_top,
                ),
            )
            rng = _rng("wordhom-voynichesque", seed, lang, t)
            src = sampler.sample(int(length * 0.75), rng)
            emit(
                "voynichesque",
                voynichesque_wordtypes_instance(
                    f"voynichesque/{lang}/t{t}",
                    _letters_to_text(src),
                    int(rng.integers(2**31)),
                    dict(truth, source_language=lang),
                    n_top=n_top,
                ),
            )
    for lang, (fam, _) in OUT_OF_INVENTORY.items():
        for t in range(per_language):
            truth = {"language": lang, "family": fam, "in_inventory": False}
            rng = _rng("wordhom-contamination", seed, lang, t)
            doc, plain = sample_out_of_inventory(lang, length, rng, root)
            emit(
                "contamination",
                wordhom_instance(
                    f"contamination/{lang}/t{t}",
                    plain,
                    targets_from_ids(plain),
                    rng,
                    dict(truth, doc=doc),
                    n_types=n_types,
                    n_top=n_top,
                ),
            )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
