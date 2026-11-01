"""OKX 칸을 바이낸스 가격에 맞추는 이동 칸 수(okx_basis_shift) -- 2026-11-02."""
from dashboard.coin_flow import okx_basis_shift


def _bars(n, okx_off, noise=0):
    base, okx = [], {}
    for i in range(n):
        base.append((i, {100: [5.0, 5.0], 101: [10.0, 10.0], 102: [5.0, 5.0]}))        # 가중 평균 칸 101
        okx[i] = {100 + okx_off + noise: [1.0, 1.0], 101 + okx_off + noise: [2.0, 2.0], 102 + okx_off + noise: [1.0, 1.0]}
    return base, okx


def test_okx_lower_by_two_buckets_shifts_up_two():
    assert okx_basis_shift(*_bars(30, -2)) == 2


def test_sub_bucket_difference_is_zero_and_no_overlap_is_zero():
    assert okx_basis_shift(*_bars(30, 0)) == 0
    assert okx_basis_shift([(1, {100: [1.0, 1.0]})], {}) == 0


if __name__ == "__main__":
    test_okx_lower_by_two_buckets_shifts_up_two(); test_sub_bucket_difference_is_zero_and_no_overlap_is_zero(); print("ok")
