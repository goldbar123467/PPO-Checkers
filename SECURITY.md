# Security policy

## Reporting

Please report a vulnerability privately through GitHub's security-advisory interface for this repository. Do not open a public issue containing credentials, private keys, live exploit details, or user data.

## Supported version

Only the current `main` branch and latest checkers policy release are supported. The website is a static demonstration: games run entirely in the browser, and there are no accounts, cookies, analytics, or server-side storage.

## Artifact safety

Verify the policy asset against its published SHA-256 before loading it. The repository uses PyTorch's weights-only loader and a closed bundle schema, but consumers should still treat unverified model files as untrusted. The website verifies the browser weights' SHA-256 against their manifest before use and ships a strict Content-Security-Policy (see `vercel.json`). Never commit `.env`, `.secrets`, W&B keys, deployment tokens, SSH private keys, datasets, full checkpoints, or run directories.
