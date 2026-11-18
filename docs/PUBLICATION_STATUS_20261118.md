# Publication progress — 2026-11-18

GitHub CLI confirmed the active `elugger` account using the local keyring. Authentication is now available; the November 13 authentication failure is historical. The target `elugger/scb-sam` did not exist when checked. No remote repository has been created and no files have been pushed.

The dependency audit identified 128 exact upstream file matches and 26 unresolved files across four inspected source families. The exact matches now have an external acquisition manifest and are excluded from Git publication. Local review copies remain. See EXTERNAL_SOURCES.md. The remaining files and other transitive components still need review before publication.

All 16 local unit tests passed after adding the source acquisition tool. Training portability and clean-environment end-to-end qualification remain incomplete. READINESS.json remains false. Source acquisition is not training verification. No new training was started.

Live acceptance: all 25 SemiSAM dependency files were subsequently downloaded through the authenticated GitHub contents API and matched the pinned byte counts and SHA-256 values. Use `--transport github-cli` with an authenticated `gh` installation if the raw download endpoint is unavailable. The other 103 exact-match files have local identity verification but were not live-downloaded in this check. All 16 tests passed again after adding the API transport.
