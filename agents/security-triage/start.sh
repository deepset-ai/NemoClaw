#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NemoClaw managed-startup entrypoint for the Security Triage agent. Modeled on
# agents/deep-research/start.sh: the agent's generator is Haystack's OpenAIChatGenerator,
# whose underlying `openai` SDK reads OPENAI_BASE_URL / OPENAI_API_KEY when the YAML leaves
# `api_base_url: null`, so NemoClaw's managed inference route (a local vLLM/NIM, or any
# OpenAI-compatible provider selected at onboarding) is injected without code changes.
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

# Managed inference: OpenAI-compatible chat completions through OpenShell's L7 proxy at
# inference.local. The real credential (if the provider has one) is injected in flight by the
# proxy, so a non-empty placeholder satisfies the SDK without embedding a secret. An operator
# who sets OPENAI_BASE_URL/OPENAI_API_KEY explicitly wins.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${NEMOCLAW_INFERENCE_BASE_URL:-https://inference.local/v1}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-nemoclaw-managed-inference}"

# The agent YAML baked into the image (security_agent.repo_scan reads SECURITY_SCAN_CONFIG). An
# operator-uploaded /sandbox/.security-triage/agent.yaml takes precedence at run time.
export SECURITY_SCAN_CONFIG="${SECURITY_SCAN_CONFIG:-/opt/security-agent-core/seeds/repo-scan.yaml}"

# The host-side security knowledge base (security-agent-core/scripts/kb_server.py). Reachable only
# if the operator applied the `security-kb` policy preset; otherwise the tool returns an error
# string and the agent carries on without it.
export SECURITY_KB_URL="${SECURITY_KB_URL:-http://host.openshell.internal:9005}"

# Route egress through the managed proxy when NemoClaw injected its coordinates. The KB call
# goes to the bridge host directly, so keep it out of the proxy.
if [ -n "${NEMOCLAW_PROXY_HOST:-}" ] && [ -n "${NEMOCLAW_PROXY_PORT:-}" ]; then
  _proxy="http://${NEMOCLAW_PROXY_HOST}:${NEMOCLAW_PROXY_PORT}"
  export HTTP_PROXY="$_proxy" HTTPS_PROXY="$_proxy" http_proxy="$_proxy" https_proxy="$_proxy"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}host.openshell.internal" no_proxy="${no_proxy:+${no_proxy},}host.openshell.internal"
  unset _proxy
fi

# Persist a credential-free env file so an on-demand scan invoked via
# `openshell sandbox exec -- security-triage-agent run ...` (a fresh process that does NOT
# inherit this entrypoint's environment) picks up the same managed inference routing + proxy.
# The wrapper CLI installed as /usr/local/bin/security-triage-agent sources it before dispatch.
{
  printf 'export HOME=%q\n' "$HOME"
  printf 'export OPENAI_BASE_URL=%q\n' "$OPENAI_BASE_URL"
  printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
  printf 'export SECURITY_SCAN_CONFIG=%q\n' "$SECURITY_SCAN_CONFIG"
  printf 'export SECURITY_KB_URL=%q\n' "$SECURITY_KB_URL"
  [ -n "${NEMOCLAW_MODEL:-}" ] && printf 'export NEMOCLAW_MODEL=%q\n' "$NEMOCLAW_MODEL"
  [ -n "${HTTP_PROXY:-}" ] && printf 'export HTTP_PROXY=%q HTTPS_PROXY=%q NO_PROXY=%q http_proxy=%q https_proxy=%q no_proxy=%q\n' \
    "$HTTP_PROXY" "$HTTPS_PROXY" "${NO_PROXY:-}" "$http_proxy" "$https_proxy" "${no_proxy:-}"
} >/tmp/nemoclaw-security-triage-env.sh 2>/dev/null || true

PORT="${NEMOCLAW_SECURITY_TRIAGE_GATEWAY_PORT:-8661}"
HOST="${NEMOCLAW_SECURITY_TRIAGE_GATEWAY_BIND:-0.0.0.0}"

# With no args this IS the sandbox's long-running entrypoint: launch the
# always-up health gateway. The actual scan is a separate, on-demand
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
