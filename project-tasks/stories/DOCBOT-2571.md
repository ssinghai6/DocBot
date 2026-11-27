# DOCBOT-2571 Microsoft sign-in (Entra OIDC)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S7 | - | 3 | senior-dev | - | To do |

## User story
As a CFO at a Microsoft 365 firm, I want to sign in with my Microsoft account, so that I do not need a separate DocBot password.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Google, GitHub and email login exist (`api/oauth_service.py`); Microsoft is only via deployment-wide SAML (`api/auth_service.py:40-60`). The Firm tier promises Microsoft sign-in.

## Scope
In:
- Multi-tenant Entra ID OIDC consumer login next to Google with PKCE.
- JIT provisioning through `resolve_provider_login` in `api/identity_service.py` (verified email only).
- Env var names `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` in `.env.example`; values set by the founder.

Out:
- Per-org SAML and domain enforcement (2572, Phase 2).

## Acceptance criteria
- [ ] Given a Microsoft work account, when the user signs in, then a user and identity row are created and they land in /chat.
- [ ] Given an existing user with the same verified email, when they sign in with Microsoft, then the identity links to that user per the DOCBOT-1617 rules.
- [ ] Given a token without a verified email claim or a tampered state, when the callback runs, then login is refused.
- [ ] Given the callback, when it completes, then an audit event is written and no token appears in logs or Sentry.

## Deliverable and approach
- Copy the Google flow structure in `oauth_service.py`; validate `iss` per tenant, `aud`, `nonce`.

## Test / verification
- Unit tests with mocked token endpoint and JWKS; founder manual login on preview.

## Dependencies
Blocked by: none
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Allow personal Microsoft accounts (consumers tenant) or work/school only?
