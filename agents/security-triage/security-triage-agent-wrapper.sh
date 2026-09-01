#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /usr/local/bin/security-triage-agent wrapper.
#
# A triage run is invoked on demand via `openshell sandbox exec --
# security-triage-agent run ...` -- a FRESH process that does not inherit the
# managed-startup entrypoint's environment (managed inference base URL,
# placeholder credentials, proxy). Source the credential-free env file that
# nemoclaw-start persisted so the exec'd run uses the same managed routing,
# then dispatch to the real venv CLI. Absent the file (e.g. a plain `docker
# run`), this is a transparent pass-through.
set -euo pipefail

if [ -f /tmp/nemoclaw-security-triage-env.sh ]; then
  # shellcheck disable=SC1091
  . /tmp/nemoclaw-security-triage-env.sh || true
fi

exec /opt/venv/bin/security-triage-agent "$@"
