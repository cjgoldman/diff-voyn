"""Cipher-head early track (CH tasks) — see
``reference_docs/Prototyping and Testing the Cipher Heads During Backbone
Training.md``.

Heads are written against the frozen :class:`~diff_voyn.heads.evaluator.Evaluator`
contract (CH.1); the n-gram DP scorer built here is the *permanent* inner-loop
scorer of design §7.4, not a throwaway mock. The diffusion evaluator plumbing
(CH.4 / task 5.1) is smoke-tested against a random-init backbone now and swaps
to the frozen EMA backbone post-G4 with no head or harness changes.
"""
