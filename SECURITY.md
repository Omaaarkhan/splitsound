# Security

## Reporting a vulnerability

Please report security problems privately through GitHub: open the
repository's **Security** tab and choose **Report a vulnerability**. Do not
open a public issue. You should get a reply within a week.

Useful things to include: the version, your operating system, what you did,
and a file that triggers the problem if there is one.

## What SplitSound trusts and what it does not

The full design is in [docs/architecture.md](docs/architecture.md#security-model).
In short:

- **Songs are untrusted.** Files are decoded with memory-safe Rust decoders,
  sizes from headers are bounded, and a file that cannot be decoded produces
  an error rather than a crash.
- **Models are pinned.** Every model the app can load is listed, with its
  SHA-256 digest, in a manifest compiled into the app. A file that does not
  match is refused. Models are plain ONNX graphs using only standard
  operators; `ml/verify_onnx.py` checks that. How each upstream checkpoint was
  obtained and verified is recorded in `ml/SECURITY.md`.
- **Pickles are never executed.** The one PyTorch checkpoint the app reads
  (the optional Demucs model, downloaded from Meta) goes through a strict
  reader that only extracts tensors and rejects anything it does not expect.
- **ONNX Runtime is Microsoft's official build**, fetched by a script that
  pins the version and the digest of each archive.
- **Updates are signed.** The app only installs an update signed with the
  project key.
- **The interface has no general access to your computer.** It runs under a
  strict Content Security Policy and can only call a fixed set of commands,
  each of which validates what it is given.
- **Nothing is uploaded.** No telemetry and no accounts. The app only contacts
  GitHub, to download models and to check for updates, and Meta's download
  server if you choose to add the optional Demucs model.

## Verifying a download

Each release includes `SHA256SUMS.txt`. After downloading an installer:

```sh
shasum -a 256 -c SHA256SUMS.txt --ignore-missing
```

Releases built from the public repository also carry GitHub build
provenance, which you can check with:

```sh
gh attestation verify <file> --repo Omaaarkhan/splitsound
```

## Supply chain

- `Cargo.lock` and `package-lock.json` are committed, and CI builds from them.
- `cargo deny` (advisories, licences, sources) and `npm audit` (the app's
  dependencies and the build tools) run on every change.
- GitHub Actions are pinned to full commit SHAs and run with the minimum token permissions.
- Release build jobs have a read-only token and no secrets; signing and
  publishing happen in a separate job.
- The update-signing key is kept in a GitHub environment that only the `main`
  branch can use. Pull-request checks rehearse the release steps with a
  throwaway key.
- Dependency updates are proposed weekly, only after a version has been public
  for at least a week, and are merged by a person, not automatically.

## Known limitations

- **The installers are not signed by Apple or Microsoft.** The project has no
  Developer ID or code-signing certificate yet, so macOS and Windows show a
  warning on first launch. Check `SHA256SUMS.txt` to be sure a download is the
  one that was published.
- **macOS library validation is off.** Without a Developer ID the app is
  signed ad hoc, and an ad-hoc app can only load the bundled ONNX Runtime
  (signed by Microsoft) with the `disable-library-validation` entitlement.
  The rest of the hardened runtime stays on. With a Developer ID the
  entitlement can be removed.
- **Windows and Linux builds are compiled and tested by CI but have had less
  hands-on use than the macOS build.**
