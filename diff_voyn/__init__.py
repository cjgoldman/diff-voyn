"""diff-voyn: language identification of Voynich-like ciphertext via trial decipherment.

Phase 0 modules (see reference_docs/Diffusion Model Training - Task Breakdown.md):

- ``diff_voyn.vocab``      — frozen vocabulary spec (task 0.1)
- ``diff_voyn.normalize``  — shared normalization pipeline (task 0.3)
- ``diff_voyn.corpus``     — corpus assembly and held-out splits (tasks 0.2, 0.4)
- ``diff_voyn.data``       — data loader / masking sampler (task 0.5)
- ``diff_voyn.infra``      — training infra: config, checkpoints, EMA, ClearML (task 0.6)
- ``diff_voyn.ciphers``    — cipher generators and controls (task 0.7)
- ``diff_voyn.vms``        — VMS transcription ingest (task 0.8)
"""

__version__ = "0.0.1"
