# Cloudflare Tunnel for Live Dashboard

This dashboard is served by `dashboard/server.py`. The server exposes the
dashboard under `/dashboard/live/` and dynamic JSON APIs under `/api/`.

## Quick tunnel

For a fast public URL using a random `trycloudflare.com` hostname:

```powershell
python .\dashboard\scripts\cloudflare_quick_tunnel.py
```

The script will:

1. Start a local dashboard server at `http://127.0.0.1:8787/dashboard/live/`
2. Download `cloudflared.exe` into `.tools/cloudflared/` if needed
3. Open a Cloudflare quick tunnel and print `PUBLIC_URL=...`

## Local-only dashboard server

If you only want the local server:

```powershell
python .\dashboard\server.py
```

Then open:

```text
http://127.0.0.1:8787/dashboard/live/
```

## Notes

- Quick tunnels are best for testing and ad-hoc access.
- If you later want a stable hostname, move to a named Cloudflare Tunnel tied to
  your account/domain.
- For `thesan.xyz`, see `dashboard/docs/cloudflare_named_tunnel.md`.
