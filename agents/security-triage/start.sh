#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NemoClaw managed-startup entrypoint for the Security Triage agent. Modeled on
# agents/deep-research/start.sh with the OPENAI_API_KEY/OPENAI_BASE_URL managed-
# inference wiring swapped for ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL (the
# `anthropic` SDK underlying AnthropicChatGenerator reads ANTHROPIC_BASE_URL
# directly when no base_url= kwarg is passed -- verified against the installed
# package; see manifest.yaml's inference comment). No Tavily/web-search
# handling: this agent has no such tool, so that whole fallback-env-var block
# from deep-research's start.sh is dropped entirely, not just left unused.
#
# Installed at /usr/local/bin/nemoclaw-start -- the FIXED path the managed-startup
# launch command runs. At `nemoclaw onboard` time OpenShell's supervisor replaces
# the image ENTRYPOINT and runs `env <profile vars> /usr/local/bin/nemoclaw-start`,
# so a health-probeable process MUST come up from this path. (A direct `openshell
# sandbox create --from <image>` instead honours the image ENTRYPOINT.)
set -euo pipefail
unset BASH_ENV ENV

export HOME=/sandbox
export PATH="/usr/local/bin:/opt/venv/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

# Managed inference: AnthropicChatGenerator (via the anthropic SDK) routes
# through OpenShell's L7 proxy at inference.local. The real credential is
# injected in flight by the proxy, so a non-empty placeholder key satisfies the
# SDK's "api key must be present" check without embedding a secret. An operator
# who points ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY elsewhere (e.g. real
# Anthropic) wins.
#
# Unlike OpenAI's convention (base URLs end in /v1, the SDK appends bare
# resource paths), the `anthropic` SDK's own default base_url
# (https://api.anthropic.com) has NO /v1 suffix -- it appends the full
# /v1/messages path itself. NEMOCLAW_INFERENCE_BASE_URL follows the
# OpenAI-style convention (used by every other agent's OpenAI-compatible
# generator) and may carry a trailing /v1; strip it here so requests don't
# resolve to a doubled /v1/v1/messages against the managed-inference proxy.
_anthropic_base_url="${ANTHROPIC_BASE_URL:-${NEMOCLAW_INFERENCE_BASE_URL:-https://inference.local}}"
export ANTHROPIC_BASE_URL="${_anthropic_base_url%/v1}"
unset _anthropic_base_url
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-nemoclaw-managed-inference}"

# Route egress through the managed proxy when NemoClaw injected its coordinates.
# This agent has exactly one egress destination (managed inference) -- no
# Tavily/web_read direct-egress carve-out is needed, so unlike deep-research's
# start.sh there is nothing to add to NO_PROXY here.
if [ -n "${NEMOCLAW_PROXY_HOST:-}" ] && [ -n "${NEMOCLAW_PROXY_PORT:-}" ]; then
  _proxy="http://${NEMOCLAW_PROXY_HOST}:${NEMOCLAW_PROXY_PORT}"
  export HTTP_PROXY="$_proxy" HTTPS_PROXY="$_proxy" http_proxy="$_proxy" https_proxy="$_proxy"
  unset _proxy
fi

# Persist a credential-free env file so an on-demand triage run invoked via
# `openshell sandbox exec -- security-triage-agent run ...` (a fresh process
# that does NOT inherit this entrypoint's environment) picks up the same
# managed inference routing + proxy. The wrapper CLI installed as
# /usr/local/bin/security-triage-agent sources it before dispatch.
{
  printf 'export HOME=%q\n' "$HOME"
  printf 'export ANTHROPIC_BASE_URL=%q\n' "$ANTHROPIC_BASE_URL"
  printf 'export ANTHROPIC_API_KEY=%q\n' "$ANTHROPIC_API_KEY"
  [ -n "${NEMOCLAW_MODEL:-}" ] && printf 'export NEMOCLAW_MODEL=%q\n' "$NEMOCLAW_MODEL"
  [ -n "${HTTP_PROXY:-}" ] && printf 'export HTTP_PROXY=%q HTTPS_PROXY=%q NO_PROXY=%q http_proxy=%q https_proxy=%q no_proxy=%q\n' \
    "$HTTP_PROXY" "$HTTPS_PROXY" "${NO_PROXY:-}" "$http_proxy" "$https_proxy" "${no_proxy:-}"
} >/tmp/nemoclaw-security-triage-env.sh 2>/dev/null || true

PORT="${NEMOCLAW_SECURITY_TRIAGE_GATEWAY_PORT:-8661}"
HOST="${NEMOCLAW_SECURITY_TRIAGE_GATEWAY_BIND:-0.0.0.0}"

# With no args this IS the sandbox's long-running entrypoint: launch the
# always-up health gateway. The actual triage run is a separate, on-demand
# `security-triage-agent run` invocation, never started here.
if [ "$#" -eq 0 ]; then
  # REQUIRED detach marker -- see agents/deep-research/start.sh's comment on
  # VM_READY_DETACH_OUTPUT_PATTERNS (src/lib/sandbox/create-stream-ready-gate.ts).
  # Must be stdout, must match /Setting up NemoClaw/.
  echo "Setting up NemoClaw Security Triage runtime..."
  echo "[nemoclaw-start] launching security-triage gateway on ${HOST}:${PORT}" >&2
  exec /usr/local/bin/security-triage-agent gateway --host "${HOST}" --port "${PORT}"
fi

exec "$@"
