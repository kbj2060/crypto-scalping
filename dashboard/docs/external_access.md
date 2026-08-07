# External Dashboard Access

This exposes the live dashboard directly on your machine without Cloudflare.

## 1. Start the dashboard server

WSL:

```bash
dashboard/scripts/start_external.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File dashboard\scripts\start_external.ps1
```

The WSL script uses the `quant_ai` conda environment by default:

```text
~/miniconda3/envs/quant_ai/bin/python
```

This starts or reuses the dashboard on:

```text
http://0.0.0.0:8787/dashboard/live/
```

Local check:

```text
http://127.0.0.1:8787/dashboard/live/
```

## 2. Open Windows Firewall

Run PowerShell as administrator:

```powershell
powershell -ExecutionPolicy Bypass -File dashboard\scripts\open_firewall.ps1
```

## 3. Router port forwarding

If you are behind a home/office router, forward external TCP port `8787` to:

```text
192.168.0.232:8787
```

For portless HTTP domain access, also forward external TCP port `80` to:

```text
192.168.0.232:80
```

Then the dashboard should be reachable from outside at:

```text
http://180.71.31.227:8787/dashboard/live/
http://180.71.31.227/dashboard/live/
```

## 4. Domain DNS

For the `thesan.xyz` domain, add these DNS records at the domain/DNS provider:

```text
Type  Name       Value
A     @          180.71.31.227
A     dashboard  180.71.31.227
```

After DNS propagation, use:

```text
http://thesan.xyz:8787/dashboard/live/
http://dashboard.thesan.xyz:8787/dashboard/live/
http://thesan.xyz/dashboard/live/
http://dashboard.thesan.xyz/dashboard/live/
```

If the DNS provider is Cloudflare and the record is proxied, disable proxying for
port `8787` or switch to a reverse proxy on port `80`/`443`.

## 5. Stop the server

WSL:

```bash
dashboard/scripts/stop_external.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File dashboard\scripts\stop_external.ps1
```

## Notes

- If your public IP changes, the outside address will change too.
- Port forwarding must be configured in your router for outside access.
- This exposes the dashboard publicly with no authentication, so use it carefully.
