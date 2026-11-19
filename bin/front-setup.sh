#!/usr/bin/env bash
# Runs on the front machine as root (bin/front deploy calls it): installs what the front needs and (re)starts it.
# Idempotent. The machine has no IPv4 address, so everything is fetched over IPv6.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
SRC=/tmp/front

if ! command -v cloudflared >/dev/null; then
  mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' > /etc/apt/sources.list.d/cloudflared.list
fi
apt-get update -qq
apt-get install -y -qq python3-venv sqlite3 cloudflared >/dev/null

if [ ! -f /swapfile ]; then                      # 1 GB of memory: a swap file keeps an install from being killed
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id so >/dev/null 2>&1 || useradd --system --home /var/lib/so --shell /usr/sbin/nologin so
install -d -o so -g so -m 700 /var/lib/so
install -d -o root -g so -m 750 /etc/so
install -d -o root -g root -m 755 /opt/so
rm -rf /opt/so/app.new && mkdir /opt/so/app.new && cp -R "$SRC/src" /opt/so/app.new/src
rm -rf /opt/so/app && mv /opt/so/app.new /opt/so/app
[ -x /opt/so/venv/bin/python ] || python3 -m venv /opt/so/venv
/opt/so/venv/bin/pip install -q --disable-pip-version-check -r /opt/so/app/src/server/front-requirements.txt
install -o root -g so -m 640 /tmp/front.env /etc/so/front.env
install -o root -g so -m 640 /tmp/front_tunnel.json /etc/so/tunnel.json
rm -f /tmp/front.env /tmp/front_tunnel.json /tmp/front.tgz
if [ -f /tmp/front.sqlite3 ] && [ ! -f /var/lib/so/server.sqlite3 ]; then      # first deploy: the users so far
  install -o so -g so -m 600 /tmp/front.sqlite3 /var/lib/so/server.sqlite3
fi
rm -f /tmp/front.sqlite3
TUNNEL_ID="$(python3 -c 'import json; print(json.load(open("/etc/so/tunnel.json"))["TunnelID"])')"

cat > /etc/systemd/system/so-front.service <<UNIT
[Unit]
Description=Shorts Pre-Publish Copilot front (sign-in, users, MCP)
After=network-online.target
Wants=network-online.target

[Service]
User=so
WorkingDirectory=/opt/so/app
EnvironmentFile=/etc/so/front.env
ExecStart=/opt/so/venv/bin/python -m src.server.app
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/so
MemoryMax=400M

[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/so-tunnel.service <<UNIT
[Unit]
Description=Cloudflare tunnel for the front
After=network-online.target
Wants=network-online.target

[Service]
User=so
ExecStart=/usr/bin/cloudflared tunnel --no-autoupdate --edge-ip-version 6 run --url http://127.0.0.1:8765 --credentials-file /etc/so/tunnel.json $TUNNEL_ID
Restart=always
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q so-front so-tunnel
systemctl restart so-front so-tunnel
sleep 4
systemctl is-active so-front so-tunnel | paste -sd' ' -
journalctl -u so-front -n 30 --no-pager -o cat | grep -E "listening|cannot start" | tail -1
