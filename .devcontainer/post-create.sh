#!/bin/bash
set -e

# Git config
git config --global --add user.email "${GIT_AUTHOR_EMAIL}"
git config --global --add user.name "${GIT_AUTHOR_NAME}"

# Persist Claude user config across rebuilds (named volume mounted at ~/.claude)
sudo chown -R vscode:vscode "$HOME/.claude" 2>/dev/null || true
mkdir -p "$HOME/.claude"

# ~/.claude.json holds auth tokens but lives outside ~/.claude; relocate + symlink
# so it's covered by the persistent volume.
if [ -f "$HOME/.claude.json" ] && [ ! -L "$HOME/.claude.json" ]; then
    mv "$HOME/.claude.json" "$HOME/.claude/.claude.json"
fi
if [ ! -e "$HOME/.claude.json" ]; then
    ln -s "$HOME/.claude/.claude.json" "$HOME/.claude.json"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Claude CLI
curl -fsSL https://claude.ai/install.sh | bash

# ---------- Python environment ----------
cd /workspace


DEVICE="${DEVICE:-cpu}"
echo "=== Setting up Python environment (${DEVICE}) ==="

if [ "$DEVICE" = "cuda" ]; then
    # Install base deps + GPU dependency group
    uv sync --group gpu

else
    # Install base deps only
    uv sync

fi



echo "=== Setup complete (${DEVICE}) ==="

# Repo-local git hooks (documentation coherence gate, scripts/doc_coherence_check.py)
git -C /workspace config core.hooksPath .githooks
