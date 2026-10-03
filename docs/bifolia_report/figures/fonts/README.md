# Voynich glyph font for the figures

`VoynichUnicode.ttf` and the EVA-to-codepoint table `EVA.TXT` are from Rebecca Bettencourt's
Voynich Unicode package (Kreative Korp, 2019; <https://github.com/kreativekorp/voynich-unicode>),
glyph designs after Gabriel Landini's EVA Hand 1 (1997) and Glen Claston's V101 (2005). The
package encodes the EVA / V101 character set in the Under-ConScript Unicode Registry (UCSUR)
private-use block U+FF400–U+FF51F; `EVA.TXT` maps each EVA transcription letter to its
codepoint. The font has no ligature table: the bench `ch`, the plumed bench `sh` and the
gallows-bench ligatures are drawn by the letter shapes joining, so an EVA string set letter by
letter renders as the manuscript's glyphs.

Used only by `fig_es_raretext.py` (executive-summary figure on rare text). The font files are
redistributed unmodified; the repository states the copyright holders (Bettencourt / Claston /
Landini) and ships the SIL Open Font License with its Fairfax builds — check the package page
before reusing the font outside this report.
