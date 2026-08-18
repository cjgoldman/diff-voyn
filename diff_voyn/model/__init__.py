"""Model package.

Phase 1: the MDLM diffusion core (task 1.1, ``diffusion``) and the
RMSNorm/SwiGLU/RoPE encoder backbone (task 1.2, ``backbone``). The Phase-0
G0 plumbing stub remains in ``stub``.
"""

from .backbone import Backbone, language_dropout_rate  # noqa: F401
from .diffusion import LN2, mdlm_loss, mdlm_nelbo_terms  # noqa: F401
from .stub import StubDenoiser  # noqa: F401
