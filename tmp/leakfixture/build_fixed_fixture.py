# 수정본 (경고 나오면 안 됨)
s1 = s0 + np.where(fired, tmin, 0)
s2 = s1 + OBS
t2u, t2d = first_touch(hi1, lo1, s2[idx], ref2 * (1 + Pv), ref2 * (1 - Pv), H * 5)
nlast = (s2[idx] - 1 - s0[idx])[:, None]
mask = (span <= nlast) & (nlast >= 0)
