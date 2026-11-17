# DNS cutover: api.docbot.space from Railway to AWS (DOCBOT-2517)

The ordering and data steps live in [../runbooks/cutover.md](../runbooks/cutover.md);
this file is the DNS part it references. Rollback: [rollback.md](rollback.md).

Facts this plan is built on:

- `docbot.space` DNS is hosted on **Vercel DNS** (nameservers `ns1/ns2.vercel-dns.com`),
  not Route 53. Terraform creates **no** DNS records; the founder edits them
  (Vercel dashboard -> Domains -> docbot.space, or `vercel dns`).
- The browser never calls the API host directly: Vercel rewrites `/api/*` to
  `BACKEND_URL` (legacy name `RAILWAY_BACKEND_URL`, read in `next.config.ts` at build time).
  So the flip is invisible to users once Vercel prod points at `api.docbot.space`.
- The prod ACM certificate is for `api.docbot.space` (`envs/prod` `api_domain`), so
  the ALB can serve that name only, and only after the cert is ISSUED.

Every step is a founder step (DNS, Railway and Vercel access). Agents prepare and verify.

## T-7d or earlier: stable hostname on Railway

1. Railway -> backend service -> Settings -> Networking -> Custom domain
   `api.docbot.space`. Railway shows a CNAME target (`<RAILWAY_CNAME_TARGET>`).
2. Vercel DNS: CNAME `api` -> `<RAILWAY_CNAME_TARGET>`.
   ```bash
   vercel dns add docbot.space api CNAME <RAILWAY_CNAME_TARGET>
   ```
3. Wait until Railway shows the domain verified with a certificate, then
   `curl -fsS https://api.docbot.space/api/health`.
4. Vercel prod env: `BACKEND_URL=https://api.docbot.space` (replace the Railway
   `*.up.railway.app` value), then **redeploy** prod (it is read at build time).
5. Check the site end to end (login, chat). Leave it like this for >= 7 days.

Also before the cutover, in `envs/prod`: the ACM validation CNAMEs from
`terraform output acm_validation_records` are added in Vercel DNS (name without the
trailing `.docbot.space.`) and **stay forever** (ACM renews through them).

## T-48h: lower TTL

The `api` record must have TTL 60 s at least 48 h before the flip, so caches holding
the Railway answer expire within a minute. Vercel DNS records default to 60 s;
confirm, and set 60 in the dashboard if not:

```bash
dig +noall +answer api.docbot.space @ns1.vercel-dns.com     # second column = TTL
```

## T+25m: test AWS before the flip

Prod has been scaled to 1 and is healthy. Hit the ALB with the real hostname (SNI
matches the cert) without touching DNS:

```bash
ALB=$(cd infra/aws/envs/prod && terraform output -raw alb_dns_name)
SMOKE_CONNECT_TO=$ALB infra/aws/smoke-test.sh https://api.docbot.space https://www.docbot.space
CONNECT_TO=$ALB infra/aws/scripts/smoke.sh https://api.docbot.space
```

## T+30m: the flip

Only after go/no-go #2 in the runbook. In the Vercel dashboard **edit** the existing
`api` CNAME value from `<RAILWAY_CNAME_TARGET>` to `<ALB_DNS_NAME>` (`terraform output
-raw alb_dns_name` in `envs/prod`). Editing in place avoids a window with no record;
the CLI can only remove and re-add (`vercel dns ls docbot.space`, `vercel dns rm <id>`,
`vercel dns add docbot.space api CNAME <ALB_DNS_NAME>`). Record the time.

## Verify on three resolvers

```bash
for r in 1.1.1.1 8.8.8.8 9.9.9.9; do echo "$r: $(dig +short api.docbot.space @$r | head -1)"; done
curl -sS -o /dev/null -w '%{http_code} %{remote_ip}\n' https://api.docbot.space/api/health
```

Done when all three return `<ALB_DNS_NAME>` (or its IPs) and health is 200. With TTL
60 this is normally within 1-2 minutes. Then run the T+40m checks in the runbook.

## After the cutover

- GitHub repo variables: `AWS_PROD_API_URL=https://api.docbot.space` (unchanged);
  **delete** `AWS_PROD_ALB_DNS` so the prod smoke test goes through DNS again.
- Keep Railway's custom domain entry until the Railway sunset (T+14d), so a rollback
  is a single CNAME edit. Railway keeps running with `WRITE_FREEZE=true`.
- TTL can go back up (e.g. 3600) after the 1-hour watch window passes with no rollback.
