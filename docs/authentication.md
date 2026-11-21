# Authentication and authorization

This note records how Open Geo Service authenticates callers and decides what they may do. The first two steps
of the [order of delivery](#order-of-delivery) are in place: named tokens, tiers and the route
table, and web interface sign-in, configured as described in
[Authentication](instance_guide.md#authentication). The rest is design.

## The principle

Open Geo Service is a resource server, not an identity provider. It never stores a password. Identity comes
from outside: named tokens for machine clients, an OIDC provider for people, and, when a DHIS2
app needs it, a trusted DHIS2 route. Authorization, ownership and audit stay inside Open Geo Service, decided
once, in one middleware, over one table of routes.

Authentication stays optional. It is on when at least one token or OIDC provider is configured,
and off otherwise:

- **Off:** no credential is checked and tiers are not enforced. Every request runs as `anonymous`,
  which only labels records. Read-only mode still applies. This is today's behaviour, and right
  for a laptop.
- **On:** a request without a valid credential is `anonymous` and gets the public tier only.

## Who calls Open Geo Service

Identity and permission are separate. A **principal** says who is calling; a **tier** says what
that principal may do. Every principal is granted exactly one tier.

| Caller | Credential | Principal id |
| --- | --- | --- |
| Anyone | none | `anonymous` |
| Machine client: CAPS, a script, a cron job | a named static token | `token:<id>` |
| Person, in the web interface or an openEO client | an OIDC token, or a session after sign-in | a local id mapped from the provider's issuer and subject |
| App inside DHIS2, when one needs it | a DHIS2 route's secret | `route:<id>` |
| The scheduler and pipeline runner | none; they run inside Open Geo Service | `service:scheduler`, `service:pipeline-runner` |

Usernames and email addresses are never principal ids: they change, and they are not unique
across providers.

## Tiers and the route table

Every method and path belongs to exactly one tier, listed in one table in the code.

| Tier | Covers |
| --- | --- |
| public | STAC, Zarr and Icechunk reads, PMTiles, thumbnails, `GET /datasets`, data sources, features, processes, workflows (`GET /process_graphs`), openEO capabilities, `/credentials/*`, the HTML pages, the API reference |
| authenticated | `POST /result`, batch jobs (`POST /jobs`, starting, listing and deleting them), storing and deleting workflows (`PUT` and `DELETE /process_graphs`) |
| operator | ingestion, sync, feature refresh, schedules, pipelines, export delivery, every `/manage` route |

Built-in and plugin workflows stay visible to everyone. A workflow stored through the API belongs
to its owner and is listed only to its owner and to operators. A stored workflow cannot take the id
of a built-in or plugin workflow unless an operator stores it. Such a workflow then replaces the
built-in for operators and for the pipelines that run it; other callers still reach the built-in.

A batch job whose `save_result` names a `dataset_id` writes into a managed dataset and publishes it,
so it needs an operator token like ingestion does. The route table cannot see an option inside the
job body, so the managed write checks the job owner's tier.

The table fails closed. A route the application serves that the table does not list, such as a
new plugin route or form endpoint, gets the operator tier, and a test over every route the
application serves fails until it is listed. The test covers routes left out of the OpenAPI
document, such as the pipeline form's lookups. A path no route serves is answered `404`.

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

Each machine client gets its own token, listed in `geoservice.yaml` with the secret in the
environment, the same way `dhis2_connections` holds DHIS2 tokens:

```yaml
authentication:
  tokens:
    - id: caps
      token_env: GEOSERVICE_CAPS_TOKEN
      tier: operator
    - id: national-sync
      token_env: GEOSERVICE_SYNC_TOKEN
      tier: operator
```

A client sends `Authorization: Bearer <secret>`. The token id is stable and not secret, and it is
written to every record the client creates, so CAPS and a script are told apart in job and
delivery history. Secrets are compared in constant time and never logged. Open Geo Service refuses to start
when two ids or two secrets repeat, or when a secret is shorter than 32 characters. The principal
appears in audit records and in structured request logs, and repeated authentication failures
are logged at a limited rate.

openEO clients use `GET /credentials/basic`, with the token id as the username and the secret as
the password. Open Geo Service checks the pair and returns the same secret as `access_token`; it mints no
second token and keeps no session. The client then sends `Authorization: Bearer basic//<secret>`,
which Open Geo Service maps to the same `token:<id>` principal as a bare `Bearer <secret>`. The openEO Python
client and Web Editor work unchanged.

Named tokens are permanent. They remain the credential for machine clients after OIDC ships.

## The web interface

The pages post plain HTML forms, which cannot send a bearer header, so the web interface has a
sign-in page. It takes a token once and sets a session cookie marked `HttpOnly`, `Secure` and
`SameSite=Lax` that holds a random session id, never the secret. Sessions are kept in memory, so a
restart signs everyone out; a session also ends after 8 hours, at sign-out, or when its token is
removed or given a new secret. An expired or unknown cookie counts as no credential and is
cleared, rather than refused, so it cannot lock the browser out of the pages.

Sign-in is offered only over HTTPS or on `localhost`, where the token does not travel in clear.
A write made with the cookie must come from the instance's own pages: its `Origin`, scheme, host
and port, must be the one the request addressed or `GEOSERVICE_BASE_URL`'s, or, without `Origin`,
`Sec-Fetch-Site` must say `same-origin`. The middleware checks this before reading the body, which a CSRF token in the
form would need. `GET /credentials/basic` answers only a `Basic` credential, so a signed-in page
cannot read the secret back.

Signed out, or signed in below the operator tier, the pages stay readable and their action panels
say they need an operator's sign-in.

The cookie serves only the Open Geo Service interface on its own origin. A client on another origin, such as a
browser app or the openEO Web Editor, sends a bearer token. Cross-site cookies, which would need
`SameSite=None`, wait until a real client needs them.

## People: OIDC

Providers are listed in configuration and advertised at `/credentials/oidc`, so openEO clients
discover them. Open Geo Service validates each token's signature against the provider's published keys, along
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

DHIS2 is an identity source, never a dependency. Open Geo Service runs without DHIS2, and signing in through
DHIS2 never implies the right to operate the service. A DHIS2 role grants an Open Geo Service tier only through an
explicit mapping.

## Apps inside DHIS2

An app running inside DHIS2 can reach Open Geo Service through a DHIS2 route without a second login. A route
is not the general way into Open Geo Service: it accepts only `GET` and `POST`, times out after at most
60 seconds and caps transfers at 5 minutes. So synchronous sync, large `POST /result` requests,
most of openEO and data delivery cannot be assumed to work through it.

Route support will be built when the first DHIS2 app names the operations it needs, and scoped to
those operations. Until then, a deployment that only DHIS2 calls can keep Open Geo Service on the internal
network, as CHAP does.

When it is built:

- The route adds a secret header through its `api-headers` authentication, and a custom DHIS2
  authority controls who may run it.
- The secret identifies the principal `route:<id>`, whose tier is set in Open Geo Service.
- Open Geo Service reads the `X-Forwarded-User` header that DHIS2 adds only after the secret matches, and
  records it as the acting person for audit. On its own the header is trivially forged and is
  ignored.
- Records created through a route belong to `route:<id>`, never to the forwarded username.

## Ownership and audit

Batch jobs and stored process graphs record the principal that created them. Listing, reading
and changing them is limited to the owner unless the caller has the operator tier, and so is
running a stored process graph by id; another principal's records are answered `404`, as if they
did not exist. A job's result files follow the same rule: they are the owner's, not anyone's
with the link. Pipeline runs and derived datasets, which no client starts, can run every stored
process graph. Records from before owners were recorded, or stored while authentication was
off, have none, and only operators reach them. `GET /me` reports the caller.

Pipeline runs, delivery records and dataset updates are reached only with the operator tier, so
ownership does not limit them. They will record the principal, how it authenticated and, where
there is one, the acting person, for audit.

The scheduler and the pipeline runner act as service principals, so an unattended delivery is as
attributable as one started by hand. The credentials Open Geo Service uses for upstream sources and DHIS2
connections are unchanged.

## Private data

Per-dataset access will use signed links, which work in QGIS, Python and browser readers alike,
where a cookie would serve only the Open Geo Service web interface. A Zarr store gets one signed path prefix,
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
   and owners on jobs and stored process graphs.
2. Web interface sign-in with a session cookie and an origin check on its writes.
3. Optional gating of reads, advertised in STAC.
4. OIDC and per-person ownership. Signing in through the provider and validating its tokens
   use an established library, such as Authlib, rather than code of our own; the session
   handling from step 2 is reconsidered then, to keep one mechanism.
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
  request headers, such as `Range`, reach Open Geo Service.
- Whether any kind of deployment should gate reads by default.
