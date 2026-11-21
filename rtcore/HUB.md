# rtcore 허브 문 단속 — Origin «같은 호스트» (2026-11-21)

`/rt` 웹소켓 업그레이드를 받는 순서는 경로 → Origin → 연결 칸이다(src/hub.rs `gate`). 이 문서는 그중 Origin 규칙만 다룬다.

## 규칙

- Origin 머리가 없으면 받는다. 브라우저가 아닌 도구(그림자 대조기·휴대폰 앱)이기 때문이다. 브라우저는 웹소켓에 Origin 을 반드시 싣는다.
- Origin 이 `RTCORE_ALLOW_ORIGINS`(쉼표 목록)의 한 항목과 바이트까지 같으면 받는다. 전과 같다.
- **더함: 같은 호스트.** Origin 의 (호스트, 포트)가 그 요청의 `Host` 머리와 같으면 목록 밖이어도 받는다(`same_host`).
  - 대소문자를 무시한다.
  - 포트가 없으면 스킴 기본값을 쓴다(https 443 · http 80). 그래서 `https://x` 와 `Host: x:443` 은 같다. 앞자리 0 도 무시한다.
  - http·https 밖의 스킴, 경로·`@`·`?`·`#` 가 섞인 값, 빈 Host, `null` 은 거짓이다. IPv6 `[..]:포트` 를 읽는다.
  - Host 가 없는데 Origin 이 있고 목록에도 없으면 403 이다.
- 그 밖은 403 `Origin` 이다.

왜 더했나: 공개 호스트 이름은 비공개 값이라 저장소·문서에 적을 수 없다. 이 규칙이 있으면 서버 환경에 목록을 따로 적지 않아도 같은 호스트에서 연 화면이 붙는다. 목록은 다른 호스트의 화면을 붙일 때만 쓴다.

CSWSH 는 계속 막힌다. 남의 사이트 페이지가 로그인된 쿠키로 열면 Origin 은 그 사이트이고 Host 는 우리 호스트다. 둘이 다르다. 브라우저는 페이지가 Origin 이나 Host 를 바꾸지 못하게 한다.

## cloudflared 가 Host 를 그대로 넘기나

넘긴다. 단 터널 원점 설정에 `httpHostHeader` 를 넣지 않았을 때만이다.

- cloudflared 소스 `ingress/origin_proxy.go`(2026-11-21 확인):
  `if o.hostHeader != "" { req.Header.Set("X-Forwarded-Host", req.Host); req.Host = o.hostHeader }`
  값이 비면 이 블록을 건너뛴다. 그러면 `req.Host` 는 들어온 그대로(공개 호스트)다.
- 공식 문서 origin-parameters 는 `httpHostHeader` 의 기본값 `""` 만 적고, 빈 값일 때 무엇이 가는지는 적지 않는다. 근거는 소스다.
- 🔴터널에 `httpHostHeader` 를 넣으면 Host 가 그 값(예 `127.0.0.1:8788`)이 된다. 그러면 같은 호스트 판정이 늘 거짓이다 → 그때는 `RTCORE_ALLOW_ORIGINS` 에 공개 Origin 을 적는다.
- 실서버에서 아직 확인하지 않았다. 확인 방법: 터널을 거친 업그레이드가 목록 없이 101 이면 Host 가 보존된 것이다.

## 시험

- tests/hub_gate.rs `g6_same_host_origin`(목록 비움): 같은 호스트 101 · 대문자 101 · 다른 호스트 403 · 포트 다름 403 · Origin 없음 101.
- tests/hub_gate.rs `same_host_rules`: 기본 포트 양쪽 · 접미 위장(`x.evil`) · 빈 Host · null · 경로 · 사용자정보 · ws 스킴 · IPv6.
- 양성 대조: `python3 rtcore/tools/mutate_hub.py`(같은 호스트 변형 8개 더함).
