# Cloudflare Named Tunnel for thesan.xyz

Use this when direct router port forwarding is unreliable.

## Cloudflare setup

1. Add `thesan.xyz` to Cloudflare, or keep Gabia DNS and create the CNAME that
   Cloudflare gives for the tunnel.
2. In Cloudflare Zero Trust, create a Tunnel.
3. Add a public hostname:

```text
Hostname: thesan.xyz
Service:  http://192.168.0.232
```

Optional:

```text
Hostname: dashboard.thesan.xyz
Service:  http://192.168.0.232
```

4. Copy the tunnel token.

## Start locally

```bash
export CLOUDFLARE_TUNNEL_TOKEN='paste-token-here'
dashboard/scripts/start_cloudflare_tunnel.sh
```

If you use the Cloudflare API, add the DNS token to `.env` and run:

```bash
dashboard/scripts/configure_cloudflare_dns.sh
```

Stop:

```bash
dashboard/scripts/stop_cloudflare_tunnel.sh
```

## Expected URLs

```text
https://thesan.xyz/dashboard/live/
https://dashboard.thesan.xyz/dashboard/live/
```
