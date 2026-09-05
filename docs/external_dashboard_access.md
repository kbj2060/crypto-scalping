# External Dashboard Access

This exposes the live dashboard directly on your machine without Cloudflare.

## 1. Start the dashboard server

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_dashboard_external.ps1
```

This starts the dashboard on:

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
powershell -ExecutionPolicy Bypass -File scripts\open_dashboard_firewall.ps1
```

## 3. Router port forwarding

If you are behind a home/office router, forward external TCP port `8787` to:

```text
192.168.1.23:8787
```

Then the dashboard should be reachable from outside at:

```text
http://222.238.86.183:8787/dashboard/live/
```

## 4. Stop the server

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_dashboard_external.ps1
```

## Notes

- If your public IP changes, the outside address will change too.
- Port forwarding must be configured in your router for outside access.
- This exposes the dashboard publicly with no authentication, so use it carefully.
