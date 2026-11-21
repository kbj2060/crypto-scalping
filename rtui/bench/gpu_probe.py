"""WSL 크로미움이 그래픽 카드를 타는지: 실행 인자 묶음마다 about:blank 에서 WebGL 렌더러 문자열을 읽는다(네트워크 없음 · 2026-11-21 결과는 README.md).
python3 -I rtui/bench/gpu_probe.py    (크로미움은 ulimit -v 를 걸면 못 뜬다 -- timeout 만)
결과 요지: 헤드리스는 어떤 인자로도 SwiftShader/llvmpipe(소프트웨어) · WSLg 창(headless=False, x11) + Mesa d3d12 만 «D3D12 (AMD Radeon RX 9060 XT)»."""
import os
from playwright.sync_api import sync_playwright

JS = """() => { const c = document.createElement('canvas'), g = c.getContext('webgl2') || c.getContext('webgl');
  if (!g) return 'WebGL 없음'; const e = g.getExtension('WEBGL_debug_renderer_info');
  return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); }"""
BASE = ["--host-resolver-rules=MAP * ~NOTFOUND", "--ignore-gpu-blocklist", "--enable-gpu"]
SETS = [  # (헤드리스?, 인자)
    (True, []), (True, ["--use-gl=angle", "--use-angle=gl"]), (True, ["--use-gl=angle", "--use-angle=gl-egl"]),
    (True, ["--use-gl=angle", "--use-angle=vulkan", "--enable-features=Vulkan"]), (True, ["--use-gl=egl"]),
    (False, ["--ozone-platform=x11"]), (False, ["--ozone-platform=x11", "--use-gl=angle", "--use-angle=gl"]),
    (False, ["--ozone-platform=wayland", "--use-gl=angle", "--use-angle=gl-egl"]),
]
env = dict(os.environ, LD_LIBRARY_PATH="/usr/lib/wsl/lib", GALLIUM_DRIVER="d3d12")   # Mesa d3d12 → /dev/dxg → 윈도 드라이버
with sync_playwright() as p:
    for head, a in SETS:
        try:
            br = p.chromium.launch(headless=head, args=BASE + a, env=env, timeout=30_000)
            try:
                print("헤드리스" if head else "WSLg 창", a, "→", br.new_page().evaluate(JS), flush=True)
            finally:
                br.close()
        except Exception as e:
            print(head, a, "실패", str(e).splitlines()[0][:150])
