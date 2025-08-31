import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import io
import datetime as dt

# ============================================================
#  基本設定 / McKinsey style
# ============================================================
st.set_page_config(page_title="経営計画策定（単年）", page_icon="📈", layout="wide")

BASE_FONT_CAND = ["Yu Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]


def set_mckinsey_style() -> None:
    """Matplotlib/Streamlit マッキンゼー風スタイル"""
    for f in BASE_FONT_CAND:
        try:
            mpl.font_manager.findfont(f, fallback_to_default=False)
            mpl.rcParams["font.family"] = f
            break
        except Exception:
            continue
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["axes.edgecolor"] = "#D0D0D0"
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.color"] = "#54565A"
    mpl.rcParams["ytick.color"] = "#54565A"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["grid.color"] = "#D0D0D0"
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["font.size"] = 11


set_mckinsey_style()

# ============================================================
#  基礎データ
# ============================================================
DEFAULTS = dict(
    sales=1_000_000_000,  # 売上高
    gp_rate=0.35,  # 粗利率
    opex_fixed=120_000_000,  # 販管費
    opex_h=170_000_000,  # 人件費
    opex_dep=60_000_000,  # 減価償却
    opex_oth=30_000_000,  # その他費用
)

BASE_PLAN = DEFAULTS.copy()

# ============================================================
#  ユーティリティ
# ============================================================

def format_money(x: float) -> str:
    return f"¥{x:,.0f}"


def format_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


# --- dual input helper ----------------------------------------------------

def _sync_pct_from_abs(
    base: float, pct_key: str, abs_key: str, src_key: str | None = None, *, margin_pt=False
):
    """abs入力からpctへ反映"""
    tgt = st.session_state.get(src_key or abs_key, base)
    st.session_state[abs_key] = tgt
    if margin_pt:
        pct = tgt - base
    else:
        pct = tgt / base - 1 if base != 0 else 0
    st.session_state[pct_key] = pct


def _sync_abs_from_pct(
    base: float, pct_key: str, abs_key: str, src_key: str | None = None, *, margin_pt=False
):
    """pct入力からabsへ反映"""
    pct = st.session_state.get(src_key or pct_key, 0.0)
    st.session_state[pct_key] = pct
    tgt = base + pct if margin_pt else base * (1 + pct)
    st.session_state[abs_key] = tgt


def dual_input(label: str, base_value: float, mode: str, pct_key: str, abs_key: str,
               kind: str = "amount", pct_range=(-0.5, 0.5)) -> float:
    """サイドバー用デュアル入力"""
    margin_pt = kind == "margin_pt"
    if pct_key not in st.session_state:
        st.session_state[pct_key] = 0.0
    if abs_key not in st.session_state:
        st.session_state[abs_key] = base_value
    col1, col2 = st.columns([1, 1])
    if mode == "％":
        st.slider(
            label,
            min_value=pct_range[0],
            max_value=pct_range[1],
            step=0.01 if margin_pt else 0.01,
            format="%" if not margin_pt else "{:.1f}",
            key=pct_key,
            on_change=_sync_abs_from_pct,
            args=(base_value, pct_key, abs_key),
            kwargs=dict(margin_pt=margin_pt),
        )
        st.number_input(
            "実額", value=st.session_state[abs_key], step=1.0,
            key=f"{abs_key}_dummy", disabled=True, format="%f")
        return st.session_state[abs_key]
    else:
        st.slider(
            label,
            min_value=max(0.0, base_value * (1 + pct_range[0])),
            max_value=base_value * (1 + pct_range[1]),
            value=st.session_state[abs_key],
            step=1.0,
            key=abs_key,
            on_change=_sync_pct_from_abs,
            args=(base_value, pct_key, abs_key),
            kwargs=dict(margin_pt=margin_pt),
        )
        st.number_input(
            "％" if not margin_pt else "pt", value=st.session_state[pct_key],
            key=f"{pct_key}_dummy", disabled=True, format="%f")
        return st.session_state[abs_key]


# --- quick slider ---------------------------------------------------------

def quick_slider(label: str, mode: str, pct_key: str, abs_key: str, base_value: float,
                 pct_range=(-0.5, 0.5), kind="amount"):
    margin_pt = kind == "margin_pt"
    if pct_key not in st.session_state:
        st.session_state[pct_key] = 0.0
    if abs_key not in st.session_state:
        st.session_state[abs_key] = float(base_value)
    if mode == "％":
        wkey = f"{pct_key}_quick"
        st.slider(
            label,
            min_value=pct_range[0],
            max_value=pct_range[1],
            value=st.session_state[pct_key],
            key=wkey,
            step=0.01,
            on_change=_sync_abs_from_pct,
            args=(base_value, pct_key, abs_key, wkey),
            kwargs=dict(margin_pt=margin_pt),
        )
    else:
        wkey = f"{abs_key}_quick"
        st.slider(
            label,
            min_value=max(0.0, base_value * (1 + pct_range[0])),
            max_value=base_value * (1 + pct_range[1]),
            value=st.session_state[abs_key],
            key=wkey,
            step=1.0,
            on_change=_sync_pct_from_abs,
            args=(base_value, pct_key, abs_key, wkey),
            kwargs=dict(margin_pt=margin_pt),
        )

# ============================================================
#  計画入力取得
# ============================================================

def collect_plan_inputs(mode: str) -> dict:
    st.sidebar.subheader("計画入力")
    sales = dual_input("売上高", BASE_PLAN['sales'], mode, 'sales_pct', 'sales_abs',
                       pct_range=(-0.5, 0.5))
    gp_rate = dual_input("粗利率(pt)", BASE_PLAN['gp_rate'], mode, 'gp_pt', 'gp_abs',
                         kind='margin_pt', pct_range=(-0.1, 0.1))
    opex_fixed = dual_input("販管費（固定費）", BASE_PLAN['opex_fixed'], mode,
                             'opex_f_pct', 'opex_f_abs')
    opex_h = dual_input("人件費", BASE_PLAN['opex_h'], mode,
                         'opex_h_pct', 'opex_h_abs')
    opex_dep = dual_input("減価償却費", BASE_PLAN['opex_dep'], mode,
                           'opex_dep_pct', 'opex_dep_abs')
    opex_oth = dual_input("その他費用", BASE_PLAN['opex_oth'], mode,
                           'opex_oth_pct', 'opex_oth_abs')
    return dict(sales=sales, gp_rate=gp_rate, opex_fixed=opex_fixed,
                opex_h=opex_h, opex_dep=opex_dep, opex_oth=opex_oth)


# ============================================================
#  計算ロジック
# ============================================================

def compute_plan(plan: dict) -> dict:
    sales = plan['sales']
    gp_rate = np.clip(plan['gp_rate'], 0, 1)
    gross = sales * gp_rate
    opex_total = plan['opex_fixed'] + plan['opex_h'] + plan['opex_dep'] + plan['opex_oth']
    op = gross - opex_total
    ord = op
    be_sales = opex_total / gp_rate if gp_rate > 0 else np.nan
    result = dict(sales=sales, gp_rate=gp_rate, gross=gross,
                  opex_fixed=plan['opex_fixed'], opex_h=plan['opex_h'],
                  opex_dep=plan['opex_dep'], opex_oth=plan['opex_oth'],
                  op=op, ord=ord, be_sales=be_sales)
    return result


# ============================================================
#  可視化
# ============================================================

def render_kpi_cards(res: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("売上高", format_money(res['sales']))
    col2.metric("粗利率", f"{res['gp_rate']*100:.1f}%")
    col3.metric("経常利益", format_money(res['ord']))
    col4.metric("BE売上", format_money(res['be_sales']))


def render_waterfall_mck(base: dict, plan: dict):
    base_res = compute_plan(base)
    plan_res = compute_plan(plan)
    steps = []
    base_op = base_res['op']
    sales_delta = (plan['sales'] - base['sales']) * base['gp_rate']
    gp_delta = plan['sales'] * (plan['gp_rate'] - base['gp_rate'])
    opex_f_delta = base['opex_fixed'] - plan['opex_fixed']
    opex_h_delta = base['opex_h'] - plan['opex_h']
    opex_dep_delta = base['opex_dep'] - plan['opex_dep']
    opex_oth_delta = base['opex_oth'] - plan['opex_oth']
    steps = [
        ("ベースOP", base_op),
        ("売上増減", sales_delta),
        ("粗利率変化", gp_delta),
        ("販管費", opex_f_delta),
        ("人件費", opex_h_delta),
        ("減価償却", opex_dep_delta),
        ("その他", opex_oth_delta),
    ]
    vals = [s[1] for s in steps]
    labels = [s[0] for s in steps]
    cum = [base_op]
    for v in vals[1:]:
        cum.append(cum[-1] + v)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#0B3D91" if v >= 0 else "#9E9E9E" for v in vals[1:]]
    ax.bar(range(1, len(vals)), vals[1:], color=colors)
    ax.axhline(0, color="#D0D0D0", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    for i, v in enumerate(vals[1:], start=1):
        ax.text(i, v + (1 if v >= 0 else -1) * max(vals) * 0.02,
                format_money(v), ha='center', va='bottom' if v >= 0 else 'top')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    st.download_button("📥 PNG保存（グラフ）", data=buf.getvalue(),
                       file_name="waterfall.png", mime="image/png")


def render_bullet_kpi(base: dict, plan: dict, target: float):
    base_res = compute_plan(base)
    plan_res = compute_plan(plan)
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(0, plan_res['ord'], color="#0B3D91", height=0.3)
    ax.barh(0, base_res['ord'], color="#9E9E9E", height=0.3)
    ax.axvline(target, color="red", linestyle="--")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax.set_xlabel("経常利益")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ============================================================
#  メインアプリ
# ============================================================

def main():
    mode = st.sidebar.radio("入力モード", ["％", "実額(円)"], horizontal=True)
    plan_inputs = collect_plan_inputs(mode)
    plan_res = compute_plan(plan_inputs)

    # メインレイアウト
    colL, colR = st.columns([6, 6], gap="large")
    with colL:
        st.subheader("クイック・コントロール")
        quick_slider("売上高", mode, 'sales_pct', 'sales_abs', BASE_PLAN['sales'])
        quick_slider("粗利率(pt)", mode, 'gp_pt', 'gp_abs', BASE_PLAN['gp_rate'],
                     kind='margin_pt', pct_range=(-0.1, 0.1))
        quick_slider("人件費", mode, 'opex_h_pct', 'opex_h_abs', BASE_PLAN['opex_h'])
        quick_slider("販管費（固定費）", mode, 'opex_f_pct', 'opex_f_abs', BASE_PLAN['opex_fixed'])

    with colR:
        st.subheader("KPI と可視化")
        render_kpi_cards(plan_res)
        render_waterfall_mck(BASE_PLAN, plan_inputs)
        render_bullet_kpi(BASE_PLAN, plan_inputs, target=BASE_PLAN['sales']*0.05)

    st.subheader("計画サマリー（表）")
    df = pd.DataFrame(
        {
            "項目": ["売上高", "粗利率", "経常利益", "BE売上"],
            "値": [format_money(plan_res['sales']), f"{plan_res['gp_rate']*100:.1f}%",
                  format_money(plan_res['ord']), format_money(plan_res['be_sales'])],
        }
    )
    st.dataframe(df, use_container_width=True)

    # Excel export
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("📊 Excel出力", data=buf.getvalue(),
                       file_name=f"plan_{dt.date.today().isoformat()}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
