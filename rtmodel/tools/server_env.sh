#!/usr/bin/env bash
# 서버와 같은 파이썬 꾸러미 판으로 정답지를 만드는 임시 venv (2026-11-21 · 서버 읽기 전용 확인: 파이썬 3.11.14 · numpy 2.2.5 · pandas 2.2.3 ·
# scikit-learn 1.8.0 · lightgbm 4.6.0 · numpy BLAS = scipy-openblas 0.3.28). PyPI 휠의 numpy 는 서버와 같은 scipy-openblas 를 묶어 온다.
#   bash rtmodel/tools/server_env.sh <venv 디렉터리>        # 한 번 만들기
#   source <(bash rtmodel/tools/server_env.sh <venv 디렉터리> --env)   # 정답지 만들 때 환경변수
# 🔴서버 CPU = Ryzen 5 5600X(Zen 3 · AVX2 · AVX-512 없음) → numpy SIMD 는 AVX-512 를 끄고, OpenBLAS 커널은 ZEN(= Haswell 커널)으로 고른다.
#   numpy 판마다 이름이 다르다: 2.4 는 X86_V4 묶음(옛 AVX512F 등은 무시) · 2.2 는 AVX512F·AVX512_SKX 등(X86_V4 없음) -- 둘 다 준다.
#   모르는 이름은 두 판 모두 오류 없이 무시하고, 함께 주면 두 판 모두 np.log 가 libm 과 20만 개 중 0개 다르다(끄지 않으면 19개 · 2026-11-21 실측).
set -euo pipefail
V="${1:?venv 디렉터리}"
if [ "${2:-}" = "--env" ]; then
  echo "export PY=$V/bin/python NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR' OPENBLAS_CORETYPE=ZEN OPENBLAS_NUM_THREADS=1"
  exit 0
fi
python3 -m venv "$V"
"$V/bin/pip" install -q "numpy==2.2.5" "pandas==2.2.3" "scikit-learn==1.8.0" "lightgbm==4.6.0" scipy pyarrow threadpoolctl joblib
"$V/bin/python" -c "import numpy, pandas, sklearn, lightgbm; print(numpy.__version__, pandas.__version__, sklearn.__version__, lightgbm.__version__)"
