# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Security Triage sandbox image (POC). Modeled on agents/deep-research/Dockerfile;
# see that file's comments for the OpenShell-supervisor / nemoclaw-start contract
# this mirrors line for line (iproute2/iptables/curl requirement, ENTRYPOINT vs
# the FIXED /usr/local/bin/nemoclaw-start managed-startup path, certifi CA symlink).
#
# Build context is the repo root: `docker build -t <ref> -f
# agents/security-triage/Dockerfile <rootDir>`.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/sandbox

# See agents/deep-research/Dockerfile for why iproute2/iptables/curl are required
# (OpenShell's in-container supervisor network-namespace setup, and the
# sandbox-exec health-probe curl NemoClaw's onboarding step runs). Unpinned to
# match the public base image's available versions, same as every sibling
# agent's Dockerfile (deep-research/hermes) -- none of them pin here either.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 \
        iptables \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root sandbox user (matches policy-additions.yaml run_as_user: sandbox).
RUN useradd --create-home --home-dir /sandbox --shell /bin/bash sandbox || true \
    && mkdir -p /sandbox/.security-triage/state /sandbox/.security-triage/logs

# Python venv at /opt/venv, matching the managed_inference binary allowlist
# (/opt/venv/bin/python3* + the kernel-resolved /usr/local/bin/python3* symlink
# target -- see policy-additions.yaml and DEEP_RESEARCH_POC_REPORT.md:158-172).
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# security-agent-core is not published to PyPI, and Docker's build context here
# is the NemoClaw repo root (no documented mechanism in this repo for pulling in
# an out-of-tree sibling package -- checked for a `docker build --build-context`
# convention and found none), so a vendored copy ships alongside this agent:
# agents/security-triage/vendor/security-agent-core/ (plain file copy, gitignored
# caches/venvs/.env excluded -- see that directory's own note if one is added).
COPY agents/security-triage/vendor/security-agent-core /opt/security-agent-core
RUN /opt/venv/bin/pip install /opt/security-agent-core

# Provision the code-triage skill's OWN venv at build time (tree-sitter/bandit),
# per its requirements.txt -- so `make_skill_script_tool(..., auto_venv=False)`
# never needs PyPI egress at run time.
RUN python3 -m venv /opt/security-agent-core/skills/code-triage/venv \
    && /opt/security-agent-core/skills/code-triage/venv/bin/pip install --upgrade pip \
    && /opt/security-agent-core/skills/code-triage/venv/bin/pip install \
        -r /opt/security-agent-core/skills/code-triage/requirements.txt

# Install the Security Triage agent runtime CLI. pyproject pulls haystack-ai==3.0.0
# + anthropic-haystack~=5.13 (security-agent-core, installed above, is what
# actually provides make_skill_script_tool). The pip entry point lands at
# /opt/venv/bin/security-triage-agent; on PATH we install a thin wrapper
# (security-triage-agent-wrapper.sh) that sources the managed env file
# nemoclaw-start persists, so `openshell sandbox exec -- security-triage-agent
# run ...` (a fresh process) still gets managed inference routing + proxy.
COPY agents/security-triage/poc-cli /opt/security-triage-agent-cli
COPY agents/security-triage/security-triage-agent-wrapper.sh /usr/local/bin/security-triage-agent
RUN /opt/venv/bin/pip install /opt/security-triage-agent-cli \
    && chmod +x /usr/local/bin/security-triage-agent

# Smoke-check the CLI is importable and on PATH at build time (via the wrapper).
RUN /usr/local/bin/security-triage-agent --version

# The OpenShell L7 proxy is TLS-terminating (MITM); the anthropic SDK (httpx-based)
# resolves CA trust via certifi.where(). Symlink certifi's bundle at SSL_CERT_FILE
# so the venv trusts the proxy CA. Mirrors agents/deep-research/Dockerfile.
ARG SSL_CERT_FILE=/etc/openshell-tls/ca-bundle.pem
# hadolint ignore=DL3059
RUN _certifi=$(/opt/venv/bin/python3 -c 'import certifi; print(certifi.where())') \
    && ln -sf "$SSL_CERT_FILE" "$_certifi"

# The managed-startup launch command runs the FIXED path /usr/local/bin/nemoclaw-start
# (not the image ENTRYPOINT, which OpenShell's supervisor replaces at onboard time).
# Also set as ENTRYPOINT so a bare `docker run` / direct `openshell sandbox create
# --from <image>` (which DOES honour ENTRYPOINT) behaves identically.
COPY agents/security-triage/start.sh /usr/local/bin/nemoclaw-start
COPY agents/security-triage/generate_config.py /opt/nemoclaw-security-triage-config/generate_config.py
RUN chmod +x /usr/local/bin/nemoclaw-start \
    && chown -R sandbox:sandbox /sandbox

EXPOSE 8661
# hadolint ignore=DL3066
USER sandbox
WORKDIR /sandbox

ENTRYPOINT ["/usr/local/bin/nemoclaw-start"]
