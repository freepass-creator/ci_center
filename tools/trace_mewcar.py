# 뮤카 원본 로고(JPEG 960×320)에서 상자·'mew'·'뮤카'를 깨끗한 벡터로 복원해 assets/mewcar/source/traced.json 에 쓴다.
# 2026-11-18 대표: 「CI 복원해서 커져도 비율 그대로」, 「직선은 직선으로 복원」.
# 방법: 윤곽을 ① 곧은 구간(최소제곱 직선, 수직·수평은 정확히 맞춤) ② 그 사이 곡선 구간(양 끝 직선 방향을 이어받는 3차 곡선, 오차가
#       크면 나눠서 다시)으로 나눈다. 짧게 뭉개진 모서리는 두 직선을 연장해 날카로운 꼭짓점으로 되살린다.
# 실행: python -I tools/trace_mewcar.py
import json, pathlib
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/mewcar/source/mewcar-logo-original.jpeg"
OUT = ROOT / "assets/mewcar/source/traced.json"
U = 10                       # 10배로 키워 윤곽을 잡는다(좌표는 원본 픽셀로 돌려 쓴다)
AXIS_DEG = 2.5               # 수직·수평에서 이 각도 안이면 정확히 수직·수평
SHORT = 3.0                  # 두 직선 사이 이보다 짧게 뭉개진 곳은 날카로운 꼭짓점으로

img = cv2.imread(str(SRC))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
cov = np.clip((hsv[..., 1] - 40) / 120, 0, 1)                      # 청록 덮임 0~1
big = cv2.GaussianBlur(cv2.resize(cov, None, fx=U, fy=U, interpolation=cv2.INTER_CUBIC), (0, 0), 0.35 * U)
teal = big > 0.5
fmt = lambda p: f"{p[0]:.3f} {p[1]:.3f}"


def cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0] if np.ndim(v) > 1 else u[0] * v[1] - u[1] * v[0]


def line_fit(pts):
    """최소제곱 직선: (한 점, 단위 방향). 수직·수평에 가까우면 정확히 맞춘다."""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy], float); c = np.array([x0, y0], float)
    ang = np.degrees(np.arctan2(d[1], d[0])) % 180
    if min(ang, 180 - ang) < AXIS_DEG:
        d = np.array([1.0, 0.0]); c = np.array([c[0], pts[:, 1].mean()])
    elif abs(ang - 90) < AXIS_DEG:
        d = np.array([0.0, 1.0]); c = np.array([pts[:, 0].mean(), c[1]])
    return c, d


def dist_line(c, d, pts):
    v = pts - c
    return np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])


def intersect(c1, d1, c2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    w = c2 - c1
    return c1 + d1 * (w[0] * d2[1] - w[1] * d2[0]) / den


def project(c, d, p):
    return c + d * np.dot(p - c, d)


def bez(P, t):
    t = t[:, None]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


def fit_cubic(Q, t0, t1):
    """Q 를 지나는 3차 곡선. 양 끝점과 방향(t0 나가는, t1 끝에서 거꾸로)을 고정하고 손잡이 길이만 맞춘다."""
    P0, P3 = Q[0], Q[-1]
    L = np.linalg.norm(P3 - P0) + 1e-9
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; u = u / (u[-1] + 1e-12)
    P = np.array([P0, P0 + t0 * L / 3, P3 + t1 * L / 3, P3])
    for _ in range(8):
        b1 = 3 * (1 - u) ** 2 * u; b2 = 3 * (1 - u) * u ** 2
        X = np.stack([b1[:, None] * t0, b2[:, None] * t1], axis=2)
        R = Q - (((1 - u) ** 3 + b1)[:, None] * P0 + (b2 + u ** 3)[:, None] * P3)
        M = np.einsum("nij,nik->jk", X, X); v = np.einsum("nij,ni->j", X, R)
        try:
            a, b = np.linalg.solve(M + 1e-6 * np.eye(2), v)
        except np.linalg.LinAlgError:
            a = b = L / 3
        a, b = np.clip(a, 0.02 * L, 1.5 * L), np.clip(b, 0.02 * L, 1.5 * L)
        P = np.array([P0, P0 + a * t0, P3 + b * t1, P3])
        B = bez(P, u)
        d1 = 3 * ((1 - u)[:, None] ** 2 * (P[1] - P[0]) + 2 * ((1 - u) * u)[:, None] * (P[2] - P[1]) + (u ** 2)[:, None] * (P[3] - P[2]))
        d2 = 6 * ((1 - u)[:, None] * (P[2] - 2 * P[1] + P[0]) + u[:, None] * (P[3] - 2 * P[2] + P[1]))
        num = np.sum((B - Q) * d1, 1); den = np.sum(d1 * d1, 1) + np.sum((B - Q) * d2, 1)
        u = np.clip(u - np.where(np.abs(den) > 1e-9, num / np.where(np.abs(den) > 1e-9, den, 1), 0), 0, 1)
        u[0], u[-1] = 0, 1; u = np.maximum.accumulate(u)
    err = np.linalg.norm(bez(P, u) - Q, axis=1)
    return P, err.max(), int(err.argmax())


def tangent_at(Q, i, k=8):
    a, b = Q[max(i - k, 0)], Q[min(i + k, len(Q) - 1)]
    v = b - a; return v / (np.linalg.norm(v) + 1e-9)


def fit_curve(Q, t0, t1, tol, depth=0):
    """오차가 tol 을 넘으면 가장 먼 점에서 나눠 둘로(이음매는 같은 방향이라 매끈하다)."""
    if len(Q) < 6:
        return [np.array([Q[0], Q[0] + (Q[-1] - Q[0]) / 3, Q[-1] - (Q[-1] - Q[0]) / 3, Q[-1]])]
    P, e, i = fit_cubic(Q, t0, t1)
    if e <= tol or depth > 5 or len(Q) < 16:
        return [P]
    i = int(np.clip(i, 6, len(Q) - 7)); tm = tangent_at(Q, i)
    return fit_curve(Q[:i + 1], t0, -tm, tol, depth + 1) + fit_curve(Q[i:], tm, t1, tol, depth + 1)


def ring(C, a, b):
    return C[a:b + 1] if b >= a else np.vstack([C[a:], C[:b + 1]])


def restore(C, min_line, tol):
    """닫힌 윤곽(원본 픽셀 좌표 n×2) → SVG 경로."""
    n = len(C)
    ap = cv2.approxPolyDP((C * U).astype(np.float32).reshape(-1, 1, 2), 0.15 * U, True)[:, 0, :] / U
    idx = sorted({int(np.argmin(np.linalg.norm(C - p, axis=1))) for p in ap})
    lines = []
    for a, b in zip(idx, idx[1:] + idx[:1]):
        seg = ring(C, a, b)
        if np.linalg.norm(seg[-1] - seg[0]) < min_line:
            continue
        k = max(1, len(seg) // 8); core = seg[k:-k] if len(seg) > 2 * k + 3 else seg
        c, d = line_fit(core)
        if dist_line(c, d, core).max() > 0.25:
            continue
        if np.dot(seg[-1] - seg[0], d) < 0:
            d = -d
        lines.append([a, b, c, d])
    # 이웃한 같은 직선 합치기
    merged = []
    for Ln in lines:
        if merged and merged[-1][1] == Ln[0] and abs(merged[-1][3][0] * Ln[3][1] - merged[-1][3][1] * Ln[3][0]) < 0.03:
            seg = ring(C, merged[-1][0], Ln[1]); c, d = line_fit(seg)
            if np.dot(seg[-1] - seg[0], d) < 0:
                d = -d
            merged[-1][1:] = [Ln[1], c, d]
        else:
            merged.append(Ln)
    lines = merged
    if not lines:   # 직선이 없는 고리: 위·오른·아래·왼 끝에서 나눠 곡선만
        ext = sorted({int(C[:, 1].argmin()), int(C[:, 0].argmax()), int(C[:, 1].argmax()), int(C[:, 0].argmin())})
        out = []
        for a, b in zip(ext, ext[1:] + ext[:1]):
            out += fit_curve(ring(C, a, b), tangent_at(C, a), -tangent_at(C, b), tol)
        return "M" + fmt(out[0][0]) + "".join("C" + fmt(P[1]) + " " + fmt(P[2]) + " " + fmt(P[3]) for P in out) + "Z"
    m = len(lines)
    pieces, first = [], None
    for i in range(m):
        A, B = lines[i], lines[(i + 1) % m]
        pA, pB = project(A[2], A[3], C[A[1]]), project(B[2], B[3], C[B[0]])
        x = intersect(A[2], A[3], B[2], B[3])
        sharp = x is not None and abs(A[3][0] * B[3][1] - A[3][1] * B[3][0]) > 0.25 \
            and np.linalg.norm(x - pA) < SHORT and np.linalg.norm(x - pB) < SHORT
        if sharp:
            pieces.append(("L", x))
        elif (B[0] - A[1]) % n <= 2 or np.linalg.norm(pB - pA) < 0.3:
            pieces.append(("L", (pA + pB) / 2))
        else:
            Q = ring(C, A[1], B[0]); Q = np.vstack([pA, Q[1:-1], pB])
            pieces.append(("L", pA)); pieces.append(("C", fit_curve(Q, A[3], -B[3], tol)))
    # 시작점 = 마지막 조각의 끝
    last = pieces[-1]
    start = last[1] if last[0] == "L" else last[1][-1][3]
    d = "M" + fmt(start)
    for k, v in pieces:
        d += ("L" + fmt(v)) if k == "L" else "".join("C" + fmt(P[1]) + " " + fmt(P[2]) + " " + fmt(P[3]) for P in v)
    return d + "Z"


# ── 꼭짓점 지정 복원(디자이너가 다시 그리듯) ──
# 모서리·곡선이 시작되는 점(앵커)을 사람이 정해 주면, 앵커 사이가 곧으면 최소제곱 직선, 굽으면 3차 곡선 하나(오차가 크면 나눔)로
# 맞춘다. 직선끼리 만나는 앵커는 두 직선의 교점(날카로운 모서리), 직선과 곡선이 매끈하게 이어지면 곡선이 직선 방향을 이어받는다.
def data_dir(C, i, step, reach=16):
    """윤곽 i 에서 step(+1/−1) 방향으로 reach 점(10점=1px) 나아가는 단위 방향."""
    n = len(C); pts = np.array([C[(i + step * k) % n] for k in range(0, reach)])
    v = pts[-1] - pts[0]
    return v / (np.linalg.norm(v) + 1e-9)


def smooth_ring(C, w):
    """닫힌 윤곽을 앞뒤 w 점 평균으로 다듬는다(JPEG 잔물결 제거, 곡선 맞춤에만 쓴다)."""
    if w <= 1:
        return C
    k = np.ones(w) / w; P = np.vstack([C[-w:], C, C[:w]])
    return np.stack([np.convolve(P[:, j], k, mode="same")[w:-w] for j in range(2)], axis=1)


def restore_anchored(C, anchors, tol, straight_tol=0.35, smooth_deg=14, reach=16, smooth=1):
    n = len(C)
    Cs = smooth_ring(C, smooth)
    idx = sorted(int(np.argmin(np.linalg.norm(C - np.array(a), axis=1))) for a in anchors)
    m = len(idx)
    segs = []
    for k in range(m):
        a, b = idx[k], idx[(k + 1) % m]
        Q = ring(C, a, b)
        ch = Q[-1] - Q[0]
        j = max(1, int(len(Q) * 0.15)); core = Q[j:-j] if len(Q) > 2 * j + 3 else Q   # 양 끝(뭉개진 모서리)은 빼고 본다
        c, d = line_fit(core)
        if dist_line(c, d, core).max() < straight_tol:
            if np.dot(ch, d) < 0:
                d = -d
            segs.append({"k": "L", "a": a, "b": b, "c": c, "d": d})
        else:
            segs.append({"k": "C", "a": a, "b": b, "Q": ring(Cs, a, b)})
    # 앵커 자리 정하기
    pos = [None] * m
    for k in range(m):
        A, B = segs[k - 1], segs[k]          # 앵커 k 는 A 의 끝, B 의 시작
        p = C[idx[k]]
        if A["k"] == B["k"] == "L":
            x = intersect(A["c"], A["d"], B["c"], B["d"])
            pos[k] = x if x is not None and np.linalg.norm(x - p) < 4 else project(B["c"], B["d"], p)
        elif A["k"] == "L":
            pos[k] = project(A["c"], A["d"], p)
        elif B["k"] == "L":
            pos[k] = project(B["c"], B["d"], p)
        else:
            pos[k] = p
    # 곡선 맞추기
    out = "M" + fmt(pos[0])
    for k in range(m):
        S = segs[k]; p0, p1 = pos[k], pos[(k + 1) % m]
        if S["k"] == "L":
            out += "L" + fmt(p1); continue
        t0 = data_dir(Cs, S["a"], +1, reach); t1 = data_dir(Cs, S["b"], -1, reach)   # t1: 끝에서 거꾸로
        prev, nxt = segs[k - 1], segs[(k + 1) % m]
        if prev["k"] == "L" and np.degrees(np.arccos(np.clip(np.dot(t0, prev["d"]), -1, 1))) < smooth_deg:
            t0 = prev["d"]
        if nxt["k"] == "L" and np.degrees(np.arccos(np.clip(np.dot(t1, -nxt["d"]), -1, 1))) < smooth_deg:
            t1 = -nxt["d"]
        Q = np.vstack([p0, S["Q"][1:-1], p1])
        for P in fit_curve(Q, t0, t1, tol):
            out += "C" + fmt(P[1]) + " " + fmt(P[2]) + " " + fmt(P[3])
    return out + "Z"


# 앵커(원본 픽셀, 대략 위치면 된다 — 가장 가까운 윤곽점으로 붙는다)
ANCHORS_MEWKA = [
    # 뮤: ㅁ 바깥(아래 두 모서리 둥금), ㅁ 구멍
    [(406.8, 6.8), (614.0, 6.8), (614.0, 134.4), (605.0, 143.4), (415.8, 143.4), (406.8, 134.4)],
    [(459.3, 47.1), (561.8, 47.1), (561.8, 103.0), (459.3, 103.0)],
    # 뮤: ㅠ (가로대 아래 모서리 둥금, 왼다리 휘어 꼬리, 오른다리 아래 오른쪽 둥금)
    [(376.8, 156.6), (643.4, 156.6), (643.4, 186.4), (633.0, 196.8), (588.0, 196.8), (588.0, 294.4), (569.0, 313.4),
     (535.2, 313.4), (535.2, 196.8), (495.6, 196.8), (409.8, 311.9), (409.8, 264.6), (446.2, 196.8), (387.8, 196.8),
     (376.8, 185.8)],
    # 카: ㅋ
    [(678.8, 6.8), (824.4, 6.8), (824.4, 150.0), (679.4, 311.7), (679.4, 271.2), (769.2, 163.6), (690.9, 163.6),
     (679.2, 151.9), (679.2, 122.8), (769.2, 122.8), (769.2, 47.4), (690.5, 47.4), (678.8, 35.7)],
    # 카: ㅏ
    [(854.2, 5.7), (907.0, 5.7), (907.0, 114.8), (952.4, 114.8), (952.4, 148.0), (938.4, 162.0), (907.0, 162.0),
     (907.0, 279.2), (872.0, 314.2), (854.2, 314.2)],
]
ANCHORS_MEW = [
    # m
    [(79.9, 77.1), (90.4, 77.1), (92.3, 78.8), (111.6, 80.4), (137.7, 91.0), (137.7, 113.8), (125.3, 113.8),
     (125.3, 95.0), (114.8, 95.0), (114.8, 113.8), (102.4, 113.8), (102.4, 95.0), (91.9, 95.0), (91.9, 113.8),
     (79.9, 113.8)],
    # e 바깥(열린 틈 포함), e 눈
    [(181.8, 99.1), (155.7, 99.7), (169.8, 102.4), (180.4, 102.3), (180.2, 104.0)],
    [(155.7, 91.2), (169.3, 91.2)],
    # w
    [(183.6, 77.0), (194.7, 77.0), (201.6, 95.2), (210.0, 77.0), (217.8, 77.0), (226.3, 95.4), (233.5, 77.0),
     (244.0, 77.0), (231.2, 113.8), (221.4, 113.8), (214.6, 96.6), (206.3, 113.8), (197.6, 113.8)],
]


def restore_set(mask, anchor_sets, tol, straight_tol, reach, smooth):
    cs = contours(mask, 4)
    out = []
    for A in anchor_sets:
        a0 = np.array(A[0]); C = min(cs, key=lambda c: np.linalg.norm(c - a0, axis=1).min())
        out.append(restore_anchored(C, A, tol, straight_tol, reach=reach, smooth=smooth))
    return "".join(out)


def contours(mask, min_area):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(float) / U + 0.5 / U for c in cs if cv2.contourArea(c) > min_area * U * U]


# 상자(왼쪽 정사각)와 '뮤카' 나누기
cols = np.where(teal.any(axis=0))[0]
gap = np.where(np.diff(cols) > 10 * U)[0][0]
bx0, bx1 = cols[0], cols[gap]
rows = np.where(teal[:, bx0:bx1 + 1].any(axis=1))[0]
by0, by1 = rows[0], rows[-1]
box = [bx0 / U, by0 / U, (bx1 + 1) / U, (by1 + 1) / U]
inner = ~teal[by0:by1 + 1, bx0:bx1 + 1]
mid = inner[:, inner.shape[1] // 4: 3 * inner.shape[1] // 4]
r = np.where(mid.mean(axis=1) > 0.01)[0]
bands, s0 = [], r[0]
for a, b in zip(r, r[1:]):
    if b - a > 2 * U:
        bands.append((s0, a)); s0 = b
bands.append((s0, r[-1]))
bands = [b for b in bands if b[1] - b[0] > 2 * U and b[0] > 4 * U]
(mew_y0, mew_y1), (bar_y0, bar_y1) = bands[0], bands[1]
bar_cols = np.where(inner[bar_y0:bar_y1 + 1].mean(axis=0) > 0.5)[0]
bar_cols = bar_cols[(bar_cols > 4 * U) & (bar_cols < inner.shape[1] - 4 * U)]

mew_mask = np.zeros_like(teal)
mew_mask[by0 + mew_y0 - 3 * U: by0 + mew_y1 + 3 * U, bx0 + 3 * U: bx1 + 1 - 3 * U] = inner[mew_y0 - 3 * U: mew_y1 + 3 * U, 3 * U: -3 * U]
word_mask = teal.copy(); word_mask[:, :bx1 + 5 * U] = False

mew_d = restore_set(mew_mask, ANCHORS_MEW, tol=0.22, straight_tol=0.22, reach=14, smooth=9)
word_d = restore_set(word_mask, ANCHORS_MEWKA, tol=0.70, straight_tol=0.45, reach=40, smooth=21)
data = {
    "source": SRC.name, "units": "원본 JPEG 픽셀(960×320)", "method": "앵커 지정 복원 — 앵커 사이 직선=최소제곱 직선(수직·수평 맞춤)·교점 모서리, 곡선=3차 곡선 맞춤",
    "box": box,
    "bar": [(bx0 + bar_cols[0]) / U, (by0 + bar_y0) / U, (bx0 + bar_cols[-1] + 1) / U, (by0 + bar_y1 + 1) / U],
    "mew": mew_d, "mewka": word_d,
}
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("box", [round(float(v), 2) for v in box], "bar", [round(float(v), 2) for v in data["bar"]],
      "mew L/C", mew_d.count("L"), mew_d.count("C"), "뮤카 L/C", word_d.count("L"), word_d.count("C"))
