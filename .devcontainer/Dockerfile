# .devcontainer/Dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Remove the existing UID 1000 user (from the base image) so that the
# devcontainers/features/common-utils feature can create the "vscode" user
# with UID 1000.
RUN if id -nu 1000 >/dev/null 2>&1; then \
        existing_user=$(id -nu 1000); \
        userdel -r "$existing_user" 2>/dev/null || userdel "$existing_user"; \
    fi

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ssh \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# LaTeX toolchain for the reports under docs/ (e.g. docs/bifolia_report: pdflatex
# via latexmk; Palatino/Helvetica/Courier, siunitx, tikz, tcolorbox, cleveref, ...).
# texlive-extra-utils brings latexindent; poppler-utils (pdftoppm/pdfinfo) and
# ghostscript are for PDF inspection and matplotlib's PDF/usetex paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-pictures \
    texlive-science \
    texlive-bibtex-extra \
    texlive-extra-utils \
    texlive-xetex \
    texlive-luatex \
    lmodern \
    cm-super \
    dvipng \
    latexmk \
    biber \
    chktex \
    lacheck \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace




