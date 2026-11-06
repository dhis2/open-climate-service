# Authentication and authorization

This note records how OCS will authenticate callers and decide what they may do. It is a design,
not a description of current behaviour: **OCS has no authentication today.** Every route is open
to anyone who can reach the instance, including ingestion, pipelines and delivery to DHIS2. Until
the first step below ships, deploy an instance privately or behind reverse-proxy authentication,
or run it read-only (see [Read-only instances](instance_guide.md#read-only-instances)).

## The principle

OCS is a resource server, not an identity provider. It never stores a password. Identity comes
from outside: named tokens for machine clients, an OIDC provider for people, and, when a DHIS2
app needs it, a trusted DHIS2 route. Authorization, ownership and audit stay inside OCS, decided
once, in one middleware, over one table of routes.

Authentication stays optional. It is on when at least one token or OIDC provider is configured,
and off otherwise:

- **Off:** no credential is checked and tiers are not enforced. Every request runs as `anonymous`,
  which only labels records. Read-only mode still applies. This is today's behaviour, and right
  for a laptop.
- **On:** a request without a valid credential is `anonymous` and gets the public tier only.

## Who calls OCS

Identity and permission are separate. A **principal** says who is calling; a **tier** says what
that principal may do. Every principal is granted exactly one tier.

| Caller | Credential | Principal id |
| --- | --- | --- |
| Anyone | none | `anonymous` |
| Machine client: CAPS, a script, a cron job | a named static token | `token:<id>` |
| Person, in the web interface or an openEO client | an OIDC token, or a session after sign-in | a local id mapped from the provider's issuer and subject |
| App inside DHIS2, when one needs it | a DHIS2 route's secret | `route:<id>` |
| The scheduler and pipeline runner | none; they run inside OCS | `service:scheduler`, `service:pipeline-runner` |

Usernames and email addresses are never principal ids: they change, and they are not unique
across providers.

## Tiers and the route table

Every method and path belongs to exactly one tier, listed in one table in the code.

| Tier | Covers |
| --- | --- |
| public | STAC, Zarr and Icechunk reads, PMTiles, thumbnails, `GET /datasets`, data sources, features, processes, workflows (`GET /process_graphs`), openEO capabilities, `/credentials/*`, the HTML pages, the API reference |
| authenticated | `POST /result`, batch jobs (`POST /jobs`, starting, listing and deleting them), storing and deleting workflows (`PUT` and `DELETE /process_graphs`) |
| operator | ingestion, sync, feature refresh, schedules, pipelines, destination delivery, every `/manage` route |

Built-in and plugin workflows stay visible to everyone. A workflow stored through the API belongs
to its owner and is listed only to its owner and to operators. A stored workflow cannot take the id
of a built-in or plugin workflow unless an operator stores it. Today such a workflow replaces the
built-in for every caller, including the pipelines that run it.

The table fails closed. A route it does not list, such as a new plugin route or form endpoint,
gets the operator tier, and a test over every route the application serves fails until it is
listed. The test covers routes left out of the OpenAPI document, such as the pipeline form's
lookups.

Read-only mode is a separate check over the same table. It refuses every route above public,
except `POST /result`, which it lets through because computing a result changes nothing. It never
changes a route's tier. With authentication on, `POST /result` still needs the authenticated
tier. Only with authentication off is it open to anyone, as public read-only instances serve the
openEO Web Editor today.

Reads stay public by default. A private deployment can gate them with an explicit setting, and
the STAC catalog then advertises the scheme through the STAC Authentication extension so STAC
Browser can offer a sign-in. Advertising a scheme does not make every reader send the credential.
Until signed links exist (see [Private data](#private-data)), gating reads can leave the map
viewer, QGIS and the Zarr, PMTiles and GeoParquet assets linked from STAC unable to load.

## Where it is enforced

One ASGI middleware decides every request before the application reads the request body. It
finds the credential, resolves the principal, applies read-only mode, and compares the
principal's tier with the route's. A missing or invalid credential gets `401` with
`WWW-Authenticate: Bearer`, and a principal without the tier gets `403`. Routes never check
credentials themselves.

The middleware sits inside CORS, so a `401` or `403` still carries the headers a browser needs to
read it, and it lets `OPTIONS` preflights through. A middleware is used rather than a FastAPI
dependency because a dependency runs after the body has been read: an unauthenticated client
could upload a large payload before its `401`, and a route added later would be unprotected
unless someone remembered the dependency.

Checks that depend on who owns a record, such as deleting a job, run in the service layer.

## Machine clients: named tokens

Each machine client gets its own token, listed in `climate-service.yaml` with the secret in the
environment, the same way `dhis2_connections` holds DHIS2 tokens:

```yaml
authentication:
  tokens:
    - id: caps
      token_env: OCS_CAPS_TOKEN
      tier: operator
    - id: national-sync
      token_env: OCS_SYNC_TOKEN
      tier: operator
```

A client sends `Authorization: Bearer <secret>`. The token id is stable and not secret, and it is
written to every record the client creates, so CAPS and a script are told apart in job and
delivery history. Secrets are compared in constant time and never logged. OCS refuses to start
when two ids or two secrets repeat, or when a secret is shorter than 32 characters. The principal
appears in audit records and in structured request logs, and repeated authentication failures
are logged at a limited rate.

openEO clients use `GET /credentials/basic`, with the token id as the username and the secret as
the password. OCS checks the pair and returns the same secret as `access_token`; it mints no
second token and keeps no session. The client then sends `Authorization: Bearer basic//<secret>`,
which OCS maps to the same `token:<id>` principal as a bare `Bearer <secret>`. The openEO Python
client and Web Editor work unchanged.

Named tokens are permanent. They remain the credential for machine clients after OIDC ships.

## The web interface

The pages post plain HTML forms, which cannot send a bearer header, so the web interface gets a
sign-in page. It sets a session cookie marked `HttpOnly`, `Secure` and `SameSite=Lax` that holds a
signed session id, never the secret. Every write made with the cookie carries a CSRF token.
Signed out, the pages stay readable and their action panels say what signing in would allow.

The cookie serves only the OCS interface on its own origin. A client on another origin, such as a
browser app or the openEO Web Editor, sends a bearer token. Cross-site cookies, which would need
`SameSite=None`, wait until a real client needs them.

## People: OIDC

Providers are listed in configuration and advertised at `/credentials/oidc`, so openEO clients
discover them. OCS validates each token's signature against the provider's published keys, along
with the issuer, the audience, the expiry and the token type. It maps the issuer and subject to a
local principal id. The operator tier comes from a claim the provider sends, or from a small local
mapping.

| Deployment | Identity |
| --- | --- |
| Laptop | none; authentication off |
| Small team, scripts, CAPS | named tokens |
| DHIS2 2.42 or later | DHIS2 as the OIDC provider, or a shared institutional provider |
| DHIS2 already behind a national provider | that provider directly |
| No DHIS2, or cloud | any OIDC provider |

DHIS2 from 2.42 is a full OIDC provider. Without a configured keystore it generates a new signing
key at every start, which invalidates every token, so a production DHIS2 used this way needs a
persistent keystore.

DHIS2 is an identity source, never a dependency. OCS runs without DHIS2, and signing in through
DHIS2 never implies the right to operate OCS. A DHIS2 role grants an OCS tier only through an
explicit mapping.

## Apps inside DHIS2

An app running inside DHIS2 can reach OCS through a DHIS2 route without a second login. A route
is not the general way into OCS: it accepts only `GET` and `POST`, times out after at most
60 seconds and caps transfers at 5 minutes. So synchronous sync, large `POST /result` requests,
most of openEO and data delivery cannot be assumed to work through it.

Route support will be built when the first DHIS2 app names the operations it needs, and scoped to
those operations. Until then, a deployment that only DHIS2 calls can keep OCS on the internal
network, as CHAP does.

When it is built:

- The route adds a secret header through its `api-headers` authentication, and a custom DHIS2
  authority controls who may run it.
- The secret identifies the principal `route:<id>`, whose tier is set in OCS.
- OCS reads the `X-Forwarded-User` header that DHIS2 adds only after the secret matches, and
  records it as the acting person for audit. On its own the header is trivially forged and is
  ignored.
- Records created through a route belong to `route:<id>`, never to the forwarded username.

## Ownership and audit

Jobs, stored process graphs, pipeline runs, delivery records and dataset updates record the
principal that created them, how it authenticated, and, where there is one, the acting person.
Listing and changing them is limited to the owner unless the caller has the operator tier.
`GET /me` reports the caller.

The scheduler and the pipeline runner act as service principals, so an unattended delivery is as
attributable as one started by hand. The credentials OCS uses for upstream sources and DHIS2
connections are unchanged.

## Private data

Per-dataset access will use signed links, which work in QGIS, Python and browser readers alike,
where a cookie would serve only the OCS web interface. A Zarr store gets one signed path prefix,
carried in the path because Zarr clients build chunk URLs by appending to the store path and drop
a query string. PMTiles, GeoParquet and thumbnails get short-lived signed URLs. This waits until
a dataset needs it.

## Limits

Authentication alone does not protect a shared server. The design also calls for a size limit on
`POST /result`, a cap on concurrent batch jobs per principal, and a rate limit on ingestion. These
apply per IP address to anonymous callers and per principal once identity exists.

## Order of delivery

Each step ships on its own and is useful alone; no later step redoes an earlier one.

1. Principals and tiers, the route table, the middleware, named tokens, the openEO basic exchange,
   and principals on records.
2. Web interface sign-in with a session cookie and CSRF.
3. Optional gating of reads, advertised in STAC.
4. OIDC and per-person ownership.
5. Role mapping and quotas.
6. Per-dataset access with signed links.
7. DHIS2 routes, for the operations the first DHIS2 app names.

## Open questions

- Which claim or mapping grants the operator tier to a person, and who administers it on a
  country instance.
- Whether headless openEO clients can sign in against DHIS2 2.42, whose documented flows are
  authorization code and refresh token, or need a provider with the device flow.
- Which DHIS2 versions the target countries run.
- For routes: whether DHIS2 replaces an `X-Forwarded-User` header the client sent, and which
  request headers, such as `Range`, reach OCS.
- Whether any kind of deployment should gate reads by default.
