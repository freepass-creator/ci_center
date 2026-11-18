# 뮤카 CI v016 — v015 디자인 그대로 순수 벡터로 복원한다(그림 파일을 SVG 에 넣지 않는다).
# 2026-11-18 대표: 「기존거를 활용」, 「뮤카는 기존거 그대로」, 영문·한글 이름은 v015 벡터 그대로.
#   - mew 상자 + '뮤카' : 원본 로고에서 복원한 윤곽(source/traced.json)을 v015 자리·크기에 맞춰 넣는다. '뮤카' 단독 CI 도 따로 낸다.
#   - 한글 '패스'·'모빌리티' : v015 벡터 그대로(대표: 「폰트가 뭐 중요하겄냐」). SUITE Heavy 시안은 MEWCAR_KO=suite 로만.
#   - 영문 'pass'·'mobility' : v015 벡터 그대로.
# 실행: python -I tools/build_mewcar.py  → assets/mewcar/v016/
import json, os, pathlib, re
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.svgLib.path import parse_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/mewcar/source"
V015 = ROOT / "assets/mewcar/v015"
OUT = pathlib.Path(os.environ.get("MEWCAR_OUT", ROOT / "assets/mewcar/v016"))
T = json.loads((SRC / "traced.json").read_text(encoding="utf-8"))
SUITE = TTFont(str(SRC / "SUITE-Heavy.woff2"))
TEAL = "#21B1BA"
VER, DATE = "v016", "20261118"

# ── v015 그림(335×108) 안의 자리. v015 PNG 를 재서 얻은 값 ──
# 2026-11-18 대표: v015 모음 그림을 보내며 「이거 그냥 벡터로 따서 만들어라 — 폰트가 뭐 중요하겄냐」. 원본 JPEG 의 mew/mobility 두 줄 상자는 «없음».
IMG_H = 108.0                       # 상자·'뮤카' 높이
BOX_W = 108.0                       # 상자 너비(정사각)
MEW_BOX = (18.0, 35.0, 88.0, 51.0)  # 상자 안 흰 'mew' 글자(왼, 위, 오른, 아래)
BAR = (18.0, 68.0, 88.0, 73.0)      # 상자 안 흰 줄


def bounds_of(d):
    bp = BoundsPen(None); parse_path(d, bp); return bp.bounds


def mewka(fg=TEAL):
    """'뮤카' 벡터를 v015 좌표(그림 왼쪽 위 0,0)에 둔다. 원본 비율 그대로 한 배율 k 로만 줄인다(「커져도 비율 그대로」)."""
    bx0, by0, bx1, by1 = T["box"]
    k = IMG_H / (by1 - by0)
    return f'<path fill="{fg}" transform="matrix({k:.6f} 0 0 {k:.6f} {-bx0 * k:.4f} {-by0 * k:.4f})" d="{T["mewka"]}"/>'


def symbol(fg=TEAL, inner="#FFFFFF"):
    """v015 의 mew 상자(정사각 + 흰 mew + 흰 밑줄). mew 글자는 원본에서 복원한 윤곽."""
    l, t, r, b = bounds_of(T["mew"])
    km = (MEW_BOX[3] - MEW_BOX[1]) / (b - t)
    cx = (MEW_BOX[0] + MEW_BOX[2]) / 2
    mew = (f'<path transform="matrix({km:.6f} 0 0 {km:.6f} {cx - (l + r) / 2 * km:.4f} {MEW_BOX[1] - t * km:.4f})" d="{T["mew"]}"/>'
           f'<rect x="{BAR[0]}" y="{BAR[1]}" width="{BAR[2] - BAR[0]}" height="{BAR[3] - BAR[1]}"/>')
    if inner in (None, "none"):   # 한 색판: 상자에서 mew·밑줄을 뚫는다(바탕이 비쳐 보인다)
        return (f'<mask id="mewcut"><rect x="0" y="0" width="{BOX_W}" height="{IMG_H}" fill="#fff"/><g fill="#000">{mew}</g></mask>'
                f'<rect x="0" y="0" width="{BOX_W}" height="{IMG_H}" fill="{fg}" mask="url(#mewcut)"/>')
    return f'<rect x="0" y="0" width="{BOX_W}" height="{IMG_H}" fill="{fg}"/><g fill="{inner}">{mew}</g>'



def mewka_left():
    bx0, by0, bx1, by1 = T["box"]
    return (bounds_of(T["mewka"])[0] - bx0) * IMG_H / (by1 - by0)


def suite_word(s):
    gs, cmap, hm = SUITE.getGlyphSet(), SUITE.getBestCmap(), SUITE["hmtx"]
    pen, bp, x = SVGPathPen(gs), BoundsPen(gs), 0.0
    for ch in s:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
        gs[g].draw(TransformPen(bp, (1, 0, 0, -1, x, 0)))
        x += hm[g][0]
    return pen.getCommands(), bp.bounds


def v015_suffix(name):
    """v015 SVG 에서 이름 그룹(변환·경로·선 굵기)을 꺼내고, 그림 안 좌표의 잉크 범위를 잰다."""
    s = (V015 / f"mewcar-{name}-v015-20261015.svg").read_text(encoding="utf-8")
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    g = re.search(r'<image [^>]+/>(<g transform="[^"]+">.*?</g>)</g></svg>', s, re.S).group(1)
    tx, ty, sc, ox, oy = map(float, re.search(r'translate\(([-\d.]+) ([-\d.]+)\) scale\(([-\d.]+)\) translate\(([-\d.]+) ([-\d.]+)\)', g).groups())
    sw = float(re.search(r'stroke-width="([\d.]+)"', g).group(1))
    L = T_ = 1e9; R = B = -1e9
    for d in re.findall(r' d="([^"]+)"', g):
        l, t, r, b = bounds_of(d); L, T_, R, B = min(L, l), min(T_, t), max(R, r), max(B, b)
    f = lambda x, y: (tx + (x + ox) * sc, ty + (y + oy) * sc)
    (l, t), (r, b) = f(L, T_), f(R, B)
    e = sw / 2 * sc
    return {"group": g, "ink": (l - e, t - e, r + e, b + e), "vb": vb}


def lockup(name, fg=TEAL, inner="#FFFFFF", bg=None):
    word, lang = name.split("-")
    v = v015_suffix(name)
    if lang == "en" or os.environ.get("MEWCAR_KO", "v015") == "v015":  # 기본: 한글도 v015 도형 그대로(MEWCAR_KO=suite 면 SUITE Heavy 시안)
        suffix = v["group"].replace(TEAL, fg)
        right = v["vb"][2] - 40
        bottom_extra = 0
    else:
        d, (l, t, r, b) = suite_word({"pass": "패스", "mobility": "모빌리티"}[word])
        il, it, ir, ib = v["ink"]
        k = (ib - it) / (b - t)                     # v015 한글 이름과 같은 높이
        suffix = f'<path fill="{fg}" transform="matrix({k:.6f} 0 0 {k:.6f} {il - l * k:.4f} {ib - b * k:.4f})" d="{d}"/>'
        right = il + (r - l) * k + (v["vb"][2] - 40 - v["ink"][2])  # v015 와 같은 오른쪽 여백
        bottom_extra = 0
    W, Hh = right + 40, v["vb"][3]
    back = f'<rect width="{W:.3f}" height="{Hh}" fill="{bg}"/>' if bg else ""
    label = {"pass-ko": "뮤카패스", "mobility-ko": "뮤카모빌리티", "pass-en": "뮤카pass", "mobility-en": "뮤카mobility"}[name]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.3f} {Hh}" width="{W:.3f}" height="{Hh}" role="img" aria-label="{label}">'
            f'<title>{label} {VER}</title>{back}<g transform="translate(40 40)">{symbol(fg, inner)}{mewka(fg)}{suffix}</g></svg>'), W


ORDER = ["pass-en", "mobility-en", "pass-ko", "mobility-ko"]
# 색: 기본(청록) · 청록 바탕(흰 글자, 상자 안 mew 는 청록) · 검정 바탕 · 흰색 한 가지 · 검정 한 가지
COLORWAYS = {"": (TEAL, "#FFFFFF", None), "-on-teal": ("#FFFFFF", TEAL, TEAL), "-on-dark": (TEAL, "#FFFFFF", "#0E1416"),
             "-white": ("#FFFFFF", None, None), "-black": ("#111111", None, None)}   # 흰·검정 한 색판은 mew 를 뚫는다


def ci_only(fg=TEAL, bg=None):
    """'뮤카' 단독 CI."""
    l, t, r, b = bounds_of(T["mewka"]); bx0, by0, bx1, by1 = T["box"]; k = IMG_H / (by1 - by0)
    x0, y0, x1, y1 = (l - bx0) * k, (t - by0) * k, (r - bx0) * k, (b - by0) * k
    pad = 0.2 * IMG_H; W, H = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    back = f'<rect width="{W:.3f}" height="{H:.3f}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.3f} {H:.3f}" width="{W:.3f}" height="{H:.3f}" role="img" aria-label="뮤카"><title>뮤카 CI {VER}</title>'
            f'{back}<g transform="translate({pad - x0:.3f} {pad - y0:.3f})">{mewka(fg)}</g></svg>')

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for cw, (fg, inner, bg) in COLORWAYS.items():
        (OUT / f"mewcar-ci{cw}-{VER}-{DATE}.svg").write_text(ci_only(fg, bg), encoding="utf-8"); made.append(f"mewcar-ci{cw}")
        for n in ORDER:
            svg, _ = lockup(n, fg, inner if inner else "none", bg)
            p = OUT / f"mewcar-{n}{cw}-{VER}-{DATE}.svg"; p.write_text(svg, encoding="utf-8"); made.append(p.name)
    # 모음(v015 모음과 같은 순서·간격 205)
    parts, wmax = [], 0
    for i, n in enumerate(ORDER):
        svg, w = lockup(n); wmax = max(wmax, w)
        inner_svg = re.search(r'<title>.*?</title>(.*)</svg>', svg, re.S).group(1)
        parts.append(f'<g transform="translate(10 {205 * i})">{inner_svg}</g>')
    fam = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {wmax + 20:.3f} 840" width="{wmax + 20:.3f}" height="840"><title>뮤카 CI 모음 {VER}</title>'
           f'<rect width="100%" height="100%" fill="white"/>{"".join(parts)}</svg>')
    (OUT / f"mewcar-family-{VER}-{DATE}.svg").write_text(fam, encoding="utf-8"); made.append(f"mewcar-family-{VER}-{DATE}.svg")
    print(len(made), "개 →", OUT)
