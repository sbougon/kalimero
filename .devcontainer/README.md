# Devcontainer

Sandboxed environment for running Codex inside this project.

## First-time setup

1. Open this directory in VS Code with the **Dev Containers** extension, then
   *Reopen in Container* - or use the CLI:
   ```sh
   devcontainer up --workspace-folder .
   devcontainer exec --workspace-folder . bash
   ```
2. Inside the container, authenticate Codex once:
   ```sh
   codex login
   ```
   Codex state is stored in the `kalimero-codex-home` named volume and persists
   across rebuilds.
3. If you need GitHub access inside Codex, authenticate GitHub once:
   ```sh
   gh auth login
   ```
   The token is stored in the `kalimero-gh-config` named volume.

## Running Codex inside the container

```sh
codex
```

The workspace is bind-mounted at `/workspace`, so edits made by Codex in the
container land in this project directory on the host.

## Timezone

The container runs on `America/Los_Angeles` (US Pacific, DST-aware) rather than
the default UTC. It is set via `ENV TZ` in the Dockerfile with `tzdata`
installed, so `date`, Node, and Codex all report Pacific time.
