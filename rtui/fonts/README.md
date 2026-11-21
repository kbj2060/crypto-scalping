# rtui 글꼴

지금 화면(`dashboard/live`)이 풋프린트 카드에서 실제로 쓰는 글꼴과 굵기 그대로다. ttf 는 저장소에 넣지 않는다(`.gitignore` `rtui/fonts/*.ttf`). 아래 주소에서 받아 이 폴더에 두고 `bash subset.sh`.

## 지금 화면이 쓰는 것 (2026-11-21 확인)

| 자리 | CSS | 굵기 |
|---|---|---|
| 차트 글자 전부(svg `<text>` · fg 캔버스 `cvText`) | `--font-sans` = "Pretendard Variable"(jsdelivr 가변 · v1.3.9) | 칸 숫자 400(`fw0` = svg 굵기, 지정 없음) · 레인 범례 400 · 시간축 700 · 봉 델타 700(`bold`) · 기둥 머리 700 · 레인 리본 이름 700 · 호버 가격 배지 700(`bold`) · 현재가 알약 800 |
| 툴팁 `#chartTooltip` | `--font-mono` = "JetBrains Mono"(구글 글꼴 wght 400~800) → 한글은 Pretendard 로 떨어짐 | 본문 400 · `<b>`(봉 시각·레인 이름) 700 · 델타·체결·TPO 조각 `font-weight:700` |

그래서 다섯 벌: Pretendard Regular·Bold·ExtraBold · JetBrains Mono Regular·Bold. 지시는 JetBrains Mono Regular 하나였지만, 툴팁이 700 도 쓴다. «화면이 다른 굵기를 쓰면 그 굵기로» 를 따라 Bold 도 넣었다(같은 배포 zip 안의 파일).
비교 하네스(`compare/harness.py`)와 측정 서버(`bench/serve.py`)는 지금 화면 쪽에 PretendardVariable.ttf 와 JetBrains Mono Regular·Bold 를 준다. 실제 화면과 같은 가변 글꼴이다. 전에는 Bold 한 벌을 모든 굵기에 줬다. 옛 방식으로 돌리려면 `RTUI_FONT_OLD=1`.

## 받은 것 (2026-11-21 · 공식 GitHub 릴리스 · 둘 다 SIL Open Font License 1.1)

| 받은 주소 | 크기 | sha256 |
|---|---|---|
| https://github.com/orioncactus/pretendard/releases/download/v1.3.9/Pretendard-1.3.9.zip | 47,304,526 | `04be351a74d6bf7d60c480a3087e51d185485d35a52023142af1df19eb8c428a` |
| https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip | 5,622,857 | `6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf` |

zip 안에서 꺼낸 파일(실행한 것 없음 · 압축만 풂):

| 파일(이 폴더) | zip 안 경로 | 크기 | sha256 |
|---|---|---|---|
| Pretendard-Regular.ttf | public/static/alternative/Pretendard-Regular.ttf | 2,725,828 | `6d0af5258997aec7354a6e340fc2325ba321c410ca48b3af858c8c3d6e92a324` |
| Pretendard-Bold.ttf | public/static/alternative/Pretendard-Bold.ttf | 2,661,752 | `c16b88c670d23e83fa1170c954cbc4822d3b8dad3c3cde15d798a94b43d97985` (11-20 부터 있던 파일과 같은 바이트) |
| Pretendard-ExtraBold.ttf | public/static/alternative/Pretendard-ExtraBold.ttf | 2,669,648 | `eedbd2877218242323bdff816684f7f5c325e54ae820d5b78eec9a5e5c7edef6` |
| PretendardVariable.ttf(하네스·측정 서버 전용 -- rtui 묶음엔 안 들어감) | public/variable/PretendardVariable.ttf | 6,739,336 | `3090ccde0442bb347aa7685d9ba8b17436a60682df6e8f92a9a670de14056e22` |
| JetBrainsMono-Regular.ttf | fonts/ttf/JetBrainsMono-Regular.ttf | 273,900 | `a0bf60ef0f83c5ed4d7a75d45838548b1f6873372dfac88f71804491898d138f` |
| JetBrainsMono-Bold.ttf | fonts/ttf/JetBrainsMono-Bold.ttf | 277,828 | `5590990c82e097397517f275f430af4546e1c45cff408bde4255dad142479dcb` |
| Pretendard-LICENSE.txt(커밋) | LICENSE.txt | 4,419 | `b04538c9abec39a3db75108cf0af0fd9c77032fe8aa2cf38345b4d250e98e38e` |
| JetBrainsMono-OFL.txt(커밋) | OFL.txt | 4,399 | `30f0c136e3c88e422d0791acd97238870f9054a9729bc34cf2ff0d4ed8cac4ad` |

## 추리기 (`subset.sh`)
- 도구: fonttools `pyftsubset`(개발기에 이미 있음).
- 남기는 글자: `rtui/src/*.rs` + `dashboard/live/app.js` + `index.html` 에 나오는 모든 글자 + ASCII + Pretendard 고정폭 숫자 U+E071~E07A. 2026-11-21 기준 1,118자(한글 완성형 포함).
- 힌팅은 뺀다. 오픈타입 기능·이름 표는 둔다.
- 추린 결과 `*.sub.ttf` 를 `src/draw.rs` 가 묶는다.
- 화면에 새 문구가 생기면 다시 돌린다. 소스에 없는 글자는 egui 기본 글꼴로 떨어진다.

## 묶음 크기 (2026-11-21)
| | wasm 원본 | wasm gzip -9 |
|---|---|---|
| 전(Pretendard-Bold 통째 한 벌 · 8d517f81) | 6,603,368 | 2,656,352 |
| 후(추린 다섯 벌 합 777,624 바이트 · acfb252e) | 4,724,778 | 1,876,279 |
