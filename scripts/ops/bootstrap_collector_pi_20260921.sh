#!/usr/bin/env bash
# 라즈베리파이 수집기(192.168.1.131) 초기 설정. Pi 위에서 pi 계정으로 실행한다.
# 멱등 -- 여러 번 돌려도 같은 결과. sshd 비밀번호 차단은 여기서 하지 않는다
# (키 로그인이 dev·server 양쪽에서 확인된 뒤 별도로 한다 -- 잠금 사고 방지).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-quant_ai}"
PY_VER="${PY_VER:-3.14}"
REPO_DIR="$HOME/crypto-scalping"
MINIFORGE="$HOME/miniforge3"

say() { printf '\n== %s\n' "$*"; }

say "시스템 패키지"
# 🔴DEBIAN_FRONTEND=noninteractive 는 debconf 만 덮고 **conffile 프롬프트는 못 막는다**.
# 2026-09-21 에 /etc/initramfs-tools/initramfs.conf 가 로컬 수정본이라 dpkg 가 Y/N 을 물었고
# stdin 이 없어 "end of file on stdin at conffile prompt" 로 죽었다. 그 한 패키지가 죽자
# 의존하는 커널 이미지 전부(linux-image-6.12.x, headers)가 딸려 실패했다.
# confnew 를 고른 이유: 이 Pi 는 USB SSD 부팅 + 물리 접근 없음이라, 부팅 실패 시 복구 수단이
# 없다. 새 initramfs.conf 의 MODULES=most 가 dep 보다 드라이버를 넓게 넣어 더 안전하다.
APT_OPTS=(-o Dpkg::Options::=--force-confnew -o Dpkg::Options::=--force-confdef)
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -qq "${APT_OPTS[@]}" full-upgrade
# rsync 는 handoff.sh 가 반드시 쓴다. 나머지는 수집기 운용 최소셋.
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -qq "${APT_OPTS[@]}" install rsync git curl ca-certificates tmux

say "시각 동기화 (수집기는 타임스탬프가 전부다)"
sudo timedatectl set-timezone Asia/Seoul
sudo timedatectl set-ntp true

# 호스트명은 바꾸지 않는다: handoff.hosts.conf 의 [collector] 별칭이 이미 이름을 주므로
# rename 은 기능이 0이고, 이 Pi 에 남아 있는 plantpoint 설정만 깨뜨릴 위험이 있다.

say "유령 ufw unit 비활성화 (바이너리 없음 + DEFAULT_INPUT_POLICY=DROP -> 재설치 시 SSH 잠금 함정)"
if ! command -v ufw >/dev/null && systemctl is-enabled ufw >/dev/null 2>&1; then
  sudo systemctl disable --now ufw || true
fi

say "miniforge + conda env '$CONDA_ENV' (handoff.sh 가 conda.sh 를 source 한다)"
if [[ ! -x "$MINIFORGE/bin/conda" ]]; then
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  bash /tmp/miniforge.sh -b -p "$MINIFORGE"
  rm -f /tmp/miniforge.sh
fi
source "$MINIFORGE/etc/profile.d/conda.sh"
conda env list | grep -qE "^${CONDA_ENV}\s" || conda create -y -q -n "$CONDA_ENV" "python=$PY_VER"
conda activate "$CONDA_ENV"
python -m pip install -q --upgrade pip
python -m pip install -q duckdb websockets aiohttp pandas pyarrow requests

say "저장소"
# --depth 1: 수집기 노드는 이력이 필요 없다. 전체 .git 은 1.9GB 라 인터넷 경유로 수십 분
# 걸린다(2026-09-21 실측). 얕은 클론이면 수십 MB.
[[ -d "$REPO_DIR/.git" ]] || git clone -q --depth 1 https://github.com/kbj2060/crypto-scalping.git "$REPO_DIR"

say "아웃바운드 키 (Pi -> dev/server 상태 체크용)"
[[ -f "$HOME/.ssh/id_ed25519" ]] || \
  ssh-keygen -q -t ed25519 -N '' -C "pi@collector (crypto-scalping)" -f "$HOME/.ssh/id_ed25519"

say "완료"
echo "hostname : $(hostname)"
echo "time     : $(timedatectl show -p Timezone -p NTPSynchronized --value | tr '\n' ' ')"
echo "python   : $(python -V)"
echo "duckdb   : $(python -c 'import duckdb;print(duckdb.__version__)')"
echo "repo     : $REPO_DIR"
echo
echo "--- Pi 공개키 (dev/server 의 authorized_keys 에 넣을 것) ---"
cat "$HOME/.ssh/id_ed25519.pub"
echo
echo "--- 저장장치 (SD 카드면 쓰기 수명 주의) ---"
lsblk -o NAME,SIZE,ROTA,TYPE,MOUNTPOINT
