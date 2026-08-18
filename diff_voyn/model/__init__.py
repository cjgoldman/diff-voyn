"""Model package. Phase 0 ships only the G0 plumbing stub; the real backbone
(RMSNorm/SwiGLU/RoPE encoder, design §3) is Phase 1 (tasks 1.1/1.2)."""

from .stub import StubDenoiser  # noqa: F401
