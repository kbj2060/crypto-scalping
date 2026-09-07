# 2026-09-08 실제 버그의 최소 재현 (스캐너 검증용 픽스처 -- 실행 안 함)
s1 = np.where(fired, s0 + np.where(fired, tmin, 0), 0)
tu2, td2 = first_touch(hi1, lo1, s1, entry * (1 + P), entry * (1 - P), H * 5)
mask = span <= tmin[idx][:, None]
