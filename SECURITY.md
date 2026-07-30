# Security policy

## Reporting

Please report a vulnerability privately through GitHub's security-advisory interface for this repository. Do not open a public issue containing credentials, private keys, live exploit details, or user data.

## Supported version

Only the current `main` branch and latest checkers policy release are supported. This is a demonstration game service with ephemeral sessions; it does not provide accounts or durable game storage.

## Artifact safety

Verify the policy asset against its published SHA-256 before loading it. The repository uses PyTorch's weights-only loader and a closed bundle schema, but consumers should still treat unverified model files as untrusted. Never commit `.env`, `.secrets`, W&B keys, Cloudflare tokens, SSH private keys, datasets, full checkpoints, or run directories.
