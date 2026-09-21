#!/usr/bin/env bash
# 라즈베리파이를 plantpoint 서버에서 수집기 전용 박스로 전환. Pi 위에서 pi 계정으로 실행.
# 사용자가 2026-09-21 에 전면 제거(도커 볼륨 포함)를 명시 승인했다. 멱등.
#
# ~/.ssh 는 절대 건드리지 않는다 -- dev/server 키가 유일한 접근 경로이고, 모니터·키보드는
# 이미 분리됐다. 여기서 authorized_keys 를 날리면 물리 작업 없이는 복구가 없다.
set -uo pipefail
say() { printf '\n== %s\n' "$*"; }

say "삭제 전 용량"
df -h / | tail -1

say "docker: 컨테이너·이미지·볼륨 전부 제거 후 패키지 purge"
if command -v docker >/dev/null; then
  sudo docker ps -aq | xargs -r sudo docker rm -f >/dev/null 2>&1
  sudo docker system prune -a --volumes -f 2>&1 | tail -1
  sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq \
    docker-ce docker-ce-cli docker-ce-rootless-extras containerd.io \
    docker-buildx-plugin docker-compose-plugin docker.io containerd runc 2>&1 | tail -2
  sudo rm -rf /var/lib/docker /var/lib/containerd /etc/docker
fi
sudo systemctl disable --now docker-compose-app gpio_controller >/dev/null 2>&1

say "헤드리스화: VNC/CUPS 제거, 콘솔 부팅"
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq \
  realvnc-vnc-server wayvnc cups cups-daemon cups-browsed 2>&1 | tail -2
# 데스크톱 패키지 자체는 남긴다: 원격에서 통째로 purge 하면 예상 밖 의존성까지 끌려가
# 부팅이 막힐 수 있다. 콘솔 부팅으로 바꾸면 RAM/CPU 이득은 거의 다 얻는다.
sudo raspi-config nonint do_boot_behaviour B1 2>/dev/null || true

say "Raspberry Pi Connect 제거 (외부 원격 접속 경로 -- 수집기엔 불필요)"
sudo systemctl --user -M pi@ disable --now rpi-connect >/dev/null 2>&1 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq rpi-connect rpi-connect-lite 2>&1 | tail -1

say "깨진 fail2ban 제거 (LAN 전용 + 키 전용 로그인이라 불필요)"
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq fail2ban 2>&1 | tail -1

say "plantpoint 및 이전 사용자 흔적 삭제 (~/.ssh 제외)"
cd "$HOME" || exit 1
for d in plantpoint plantpoint-automation .pm2 .npm .npm-global .cursor .cursor-server \
         .claude .claude.json .claude.json.backup .docker .env .git-credentials \
         .bash_history .python_history .lesshst .logs .vim .cache .config/lxsession; do
  [[ "$d" == ".ssh"* ]] && continue   # 방어: 절대 삭제하지 않는다
  [[ -e "$HOME/$d" ]] && { rm -rf "${HOME:?}/$d" && echo "  삭제: ~/$d"; }
done

say "apt 정리"
sudo DEBIAN_FRONTEND=noninteractive apt-get autoremove --purge -y -qq 2>&1 | tail -1
sudo apt-get clean
# rc(설정만 남은) 패키지 잔재 정리
dpkg -l | awk '/^rc/{print $2}' | xargs -r sudo dpkg --purge >/dev/null 2>&1

say "결과"
echo "접근 경로 확인 (이게 살아 있어야 한다):"
ls -l "$HOME/.ssh/authorized_keys" && cut -d' ' -f3- "$HOME/.ssh/authorized_keys" | sed 's/^/  /'
echo "listening:"; sudo ss -ltnp 2>/dev/null | tail -n +2 | awk '{print "  "$4}'
echo "깨진 패키지:"; dpkg -l | awk 'NR>5 && $1 !~ /^(ii|rc)/ {print "  "$1, $2}' || true
df -h / | tail -1
