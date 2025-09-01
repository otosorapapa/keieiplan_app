import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import io
import datetime as dt
import os
from openai import OpenAI

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
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["grid.color"] = "#D0D0D0"
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["font.size"] = 11


set_mckinsey_style()

# ------------------------------------------------------------
#  Custom CSS for light/dark mode and spacing
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --base-bg: #F5F5F5;
        --base-text: #000000;
        --primary: #0B3D91;
    }
    [data-theme="dark"] {
        --base-bg: #1E1E1E;
        --base-text: #FFFFFF;
    }
    html, body, [class*="stApp"] {
        background-color: var(--base-bg);
        color: var(--base-text);
    }
    .stMetric {
        background-color: rgba(11, 61, 145, 0.1);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 0.5rem;
        color: var(--base-text);
    }
    [data-theme="dark"] .stMetric {
        background-color: rgba(11, 61, 145, 0.3);
    }
    .stMetric label {
        color: var(--primary);
    }
    .stButton>button {
        background-color: var(--primary);
        color: #FFFFFF;
    }
    .stPlotlyChart, .stDataFrame, .stMetric, .stButton {
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
#  Generative AI helpers
# ============================================================


def _openai_generate(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY が設定されていません。環境変数に API キーをセットしてください。"
    client = OpenAI(api_key=api_key)
    try:
        completion = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return completion.output_text.strip()
    except Exception as e:
        return f"AI生成に失敗しました: {e}"


def generate_ai_summary(res: dict) -> str:
    """KPI要約を3行以内で生成"""
    prompt = (
        "あなたは経営コンサルタントです。以下の経営計画データを基に、"
        "主要KPIのポイントを3行以内で簡潔に日本語で要約してください。\n"
        f"売上高: {res['sales']:,} 円\n"
        f"粗利率: {res['gp_rate']*100:.1f}%\n"
        f"経常利益: {res['ord']:,} 円\n"
        f"損益分岐点売上: {res['be_sales']:,} 円"
    )
    return _openai_generate(prompt)


def generate_ai_comments(res: dict) -> str:
    """改善余地やリスク要因のコメントを生成"""
    prompt = (
        "あなたは経営コンサルタントです。以下の経営計画データを基に、"
        "改善余地やリスク要因を1〜2行でコメントしてください。\n"
        f"売上高: {res['sales']:,} 円\n"
        f"粗利率: {res['gp_rate']*100:.1f}%\n"
        f"経常利益: {res['ord']:,} 円\n"
        f"損益分岐点売上: {res['be_sales']:,} 円"
    )
    return _openai_generate(prompt)


def generate_ai_explanation(res: dict) -> str:
    """損益分岐点や利益構造を初心者向けに解説"""
    prompt = (
        "あなたは優しい経営コンサルタントです。以下の経営計画データを基に、"
        "損益分岐点や利益構造を初心者でもわかるように簡単に説明してください。\n"
        f"売上高: {res['sales']:,} 円\n"
        f"粗利率: {res['gp_rate']*100:.1f}%\n"
        f"経常利益: {res['ord']:,} 円\n"
        f"損益分岐点売上: {res['be_sales']:,} 円"
    )
    return _openai_generate(prompt)


# ============================================================
#  可視化
# ============================================================


def build_chart_style_from_sidebar() -> dict:
    st.sidebar.subheader("グラフ表示のカスタマイズ")
    with st.sidebar.expander("🎨 配色・表示設定", expanded=False):
        fig_bg = st.color_picker("図の背景色 (figure)", "#FFFFFF")
        ax_bg = st.color_picker("軸の背景色 (axes)", "#FFFFFF")
        grid_on = st.checkbox("グリッドを表示", True)
        grid_color = st.color_picker("グリッド色", "#D0D0D0")
        font_color = st.color_picker("フォント色", "#222222")
        font_size = st.slider("フォントサイズ", 8, 18, 11, 1)
        alpha = st.slider("透過度 (alpha)", 10, 100, 100, 1) / 100.0

    with st.sidebar.expander("🧱 バー／線／ノードの設定", expanded=False):
        pos_color = st.color_picker("正の値の色", "#0B3D91")
        neg_color = st.color_picker("負の値の色", "#9E9E9E")
        line_color = st.color_picker("線の色（バレット指標線など）", "#FF0000")
        marker_size = st.slider("ノード/マーカーサイズ", 4, 20, 8, 1)
        bar_width = st.slider("バーの太さ（相対）", 1, 100, 100, 1) / 100.0

    return dict(
        fig_bg=fig_bg,
        ax_bg=ax_bg,
        grid_on=grid_on,
        grid_color=grid_color,
        font_color=font_color,
        font_size=font_size,
        alpha=alpha,
        pos_color=pos_color,
        neg_color=neg_color,
        line_color=line_color,
        marker_size=marker_size,
        bar_width=bar_width,
    )


def render_kpi_cards(res: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("売上高", format_money(res['sales']))
    col2.metric("粗利率", f"{res['gp_rate']*100:.1f}%")
    col3.metric("経常利益", format_money(res['ord']))
    col4.metric("BE売上", format_money(res['be_sales']))


def render_waterfall_mck(base: dict, plan: dict, style: dict):
    base_res = compute_plan(base)
    plan_res = compute_plan(plan)
    base_op = base_res['op']
    sales_delta = (plan['sales'] - base['sales']) * base['gp_rate']
    gp_delta = plan['sales'] * (plan['gp_rate'] - base['gp_rate'])
    opex_f_delta = base['opex_fixed'] - plan['opex_fixed']
    opex_h_delta = base['opex_h'] - plan['opex_h']
    opex_dep_delta = base['opex_dep'] - plan['opex_dep']
    opex_oth_delta = base['opex_oth'] - plan['opex_oth']

    labels = ["ベースOP", "売上増減", "粗利率変化", "販管費", "人件費", "減価償却", "その他"]
    vals = [base_op, sales_delta, gp_delta, opex_f_delta, opex_h_delta, opex_dep_delta, opex_oth_delta]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(style["fig_bg"])
    ax.set_facecolor(style["ax_bg"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=style["font_color"], labelsize=style["font_size"])
    ax.yaxis.label.set_color(style["font_color"])
    ax.xaxis.label.set_color(style["font_color"])

    ax.grid(style["grid_on"], color=style["grid_color"], linewidth=0.6, axis="y")

    colors = []
    for i, v in enumerate(vals):
        if i == 0:
            colors.append(style["pos_color"])
        else:
            colors.append(style["pos_color"] if v >= 0 else style["neg_color"])

    ax.bar(range(len(vals)), vals, color=colors, alpha=style["alpha"], width=0.8 * style["bar_width"])
    ax.axhline(0, color=style["grid_color"], linewidth=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=style["font_color"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

    offset = max(abs(x) for x in vals) * 0.02 if any(vals) else 1.0
    for i, v in enumerate(vals):
        ax.plot(i, v, marker="o", markersize=style["marker_size"], color=style["font_color"], alpha=0.9)
        ax.text(
            i,
            v + (offset if v >= 0 else -offset),
            format_money(v),
            ha="center",
            va="bottom" if v >= 0 else "top",
            color=style["font_color"],
            fontsize=style["font_size"],
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button(
        "📥 PNG保存（グラフ）",
        data=buf.getvalue(),
        file_name="waterfall.png",
        mime="image/png",
    )


def render_bullet_kpi(base: dict, plan: dict, target: float, style: dict):
    base_res = compute_plan(base)
    plan_res = compute_plan(plan)
    fig, ax = plt.subplots(figsize=(6, 1.2))

    fig.patch.set_facecolor(style["fig_bg"])
    ax.set_facecolor(style["ax_bg"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=style["font_color"], labelsize=style["font_size"])
    ax.grid(style["grid_on"], color=style["grid_color"], linewidth=0.6, axis="x")

    ax.barh(0, plan_res['ord'], color=style["pos_color"], height=0.3, alpha=style["alpha"])
    ax.barh(0, base_res['ord'], color=style["neg_color"], height=0.3, alpha=style["alpha"])
    ax.axvline(target, color=style["line_color"], linestyle="--", linewidth=2)

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax.set_xlabel("経常利益", color=style["font_color"], fontsize=style["font_size"])

    ax.plot(target, 0, marker="o", markersize=style["marker_size"], color=style["line_color"])

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_profit_gauge(res: dict, target: float, style: dict) -> None:
    st.subheader("利益達成度")
    progress = res['ord'] / target if target else 0
    progress = float(np.clip(progress, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(6, 0.6))
    fig.patch.set_facecolor(style["fig_bg"])
    ax.set_facecolor(style["ax_bg"])
    ax.barh(0, 1.0, color=style["neg_color"], alpha=style["alpha"])
    ax.barh(0, progress, color=style["pos_color"], alpha=style["alpha"])
    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    if res['ord'] >= target:
        st.success(f"目標利益 {format_money(target)} を達成しています！")
    else:
        st.info(f"目標まであと {format_money(target - res['ord'])}")


def build_viz_controls():
    st.subheader("📈 ビジュアライズ")
    col1, col2 = st.columns([2, 1])
    with col1:
        viz_type = st.radio(
            "表示グラフを選択",
            ["ウォーターフォール", "トルネード（感応度）", "コスト構成スタック", "P/Lブリッジ（ベース→計画）"],
            horizontal=True,
        )
    with col2:
        sens_step = st.slider("感応度ステップ（±）", 0.01, 0.20, 0.10, 0.01)
    return viz_type, sens_step


def render_tornado_sensitivity(base: dict, plan: dict, step: float = 0.1):
    """指標：経常利益に対する各変数の影響度（±step の差分）"""
    keys = [
        ("sales", "売上高", "amount"),
        ("gp_rate", "粗利率", "rate"),
        ("opex_h", "人件費", "amount"),
        ("opex_fixed", "販管費（固定費）", "amount"),
        ("opex_dep", "減価償却", "amount"),
        ("opex_oth", "その他費用", "amount"),
    ]
    base_res = compute_plan(plan)
    base_ord = base_res["ord"]
    items, lows, highs = [], [], []

    for k, label, kind in keys:
        p_low = plan.copy()
        p_high = plan.copy()
        if kind == "rate":
            p_low[k] = max(0.0, plan[k] - step)
            p_high[k] = min(1.0, plan[k] + step)
        else:
            p_low[k] = max(0.0, plan[k] * (1 - step))
            p_high[k] = plan[k] * (1 + step)

        low_ord = compute_plan(p_low)["ord"]
        high_ord = compute_plan(p_high)["ord"]
        items.append(label)
        lows.append(low_ord - base_ord)
        highs.append(high_ord - base_ord)

    deltas = [(min(l, h), max(l, h)) for l, h in zip(lows, highs)]
    ranges = [abs(lo) + abs(hi) for lo, hi in deltas]
    order = np.argsort(ranges)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, i in enumerate(order):
        lo, hi = deltas[i]
        ax.barh(idx, hi, color="#0B3D91")
        ax.barh(idx, lo, color="#9E9E9E")
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels([items[i] for i in order])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax.axvline(0, color="#D0D0D0", linewidth=0.8)
    ax.set_xlabel("経常利益への寄与（差分）")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_cost_stack(plan: dict):
    """粗利・各費用の構成をスタック棒で表示"""
    res = compute_plan(plan)
    labels = ["粗利", "人件費", "販管費（固定）", "減価償却", "その他費用"]
    vals = [
        res["gross"],
        res["opex_h"],
        res["opex_fixed"],
        res["opex_dep"],
        res["opex_oth"],
    ]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(6, 1.5))
    left = 0.0
    for v in vals:
        ax.barh(0, v, left=left, height=0.5)
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, max(total * 1.05, 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax.set_xlabel("金額（構成比）")
    handles = [plt.Rectangle((0, 0), 1, 1) for _ in vals]
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_pl_bridge(base: dict, plan: dict):
    """ベース→計画のP/Lブリッジ（OPの増減要因）"""
    base_res = compute_plan(base)
    plan_res = compute_plan(plan)
    steps = [
        ("ベースOP", base_res["op"]),
        ("売上×ベース粗利率", (plan["sales"] - base["sales"]) * base_res["gp_rate"]),
        ("粗利率変化", plan["sales"] * (plan["gp_rate"] - base_res["gp_rate"])),
        ("人件費Δ", base["opex_h"] - plan["opex_h"]),
        ("販管費Δ", base["opex_fixed"] - plan["opex_fixed"]),
        ("減価償却Δ", base["opex_dep"] - plan["opex_dep"]),
        ("その他Δ", base["opex_oth"] - plan["opex_oth"]),
    ]
    labels = [s[0] for s in steps]
    vals = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#0B3D91" if v >= 0 else "#9E9E9E" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="#D0D0D0", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + (1 if v >= 0 else -1) * max(vals, key=abs) * 0.02,
            f"{format_money(v)}",
            ha="center",
            va="bottom" if v >= 0 else "top",
        )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_visualizations(viz_type: str, sens_step: float, base: dict, plan: dict, style: dict):
    """選択に応じたグラフ描画の統括"""
    if viz_type == "ウォーターフォール":
        render_waterfall_mck(base, plan, style)
    elif viz_type == "トルネード（感応度）":
        render_tornado_sensitivity(base, plan, sens_step)
    elif viz_type == "コスト構成スタック":
        render_cost_stack(plan)
    else:
        render_pl_bridge(base, plan)


# ============================================================
#  メインアプリ
# ============================================================

def main():
    mode = st.sidebar.radio("入力モード", ["％", "実額(円)"], horizontal=True)
    plan_inputs = collect_plan_inputs(mode)
    plan_res = compute_plan(plan_inputs)

    style = build_chart_style_from_sidebar()

    colL, colR = st.columns([6, 6], gap="large")
    with colL:
        st.subheader("🎛️ クイック・コントロール")
        quick_slider("売上高", mode, 'sales_pct', 'sales_abs', BASE_PLAN['sales'])
        quick_slider("粗利率(pt)", mode, 'gp_pt', 'gp_abs', BASE_PLAN['gp_rate'],
                     kind='margin_pt', pct_range=(-0.1, 0.1))
        quick_slider("人件費", mode, 'opex_h_pct', 'opex_h_abs', BASE_PLAN['opex_h'])
        quick_slider("販管費（固定費）", mode, 'opex_f_pct', 'opex_f_abs', BASE_PLAN['opex_fixed'])

    with colR:
        with st.container():
            st.subheader("📊 KPI と可視化")
            render_kpi_cards(plan_res)
            viz_type, sens_step = build_viz_controls()
            render_visualizations(viz_type, sens_step, BASE_PLAN, plan_inputs, style)
            target_ord = BASE_PLAN['sales'] * 0.05
            render_bullet_kpi(BASE_PLAN, plan_inputs, target=target_ord, style=style)
            render_profit_gauge(plan_res, target=target_ord, style=style)

    with st.container():
        st.subheader("📝 計画サマリー")
        df = pd.DataFrame(
            {
                "項目": ["売上高", "粗利率", "経常利益", "BE売上"],
                "値": [format_money(plan_res['sales']), f"{plan_res['gp_rate']*100:.1f}%",
                      format_money(plan_res['ord']), format_money(plan_res['be_sales'])],
            }
        )
        st.dataframe(df, use_container_width=True)

    with st.container():
        st.subheader("🤖 AIサマリー")
        st.write(generate_ai_summary(plan_res))
        st.write(generate_ai_comments(plan_res))
        st.write(generate_ai_explanation(plan_res))

    # Excel export
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("📊 Excel出力", data=buf.getvalue(),
                       file_name=f"plan_{dt.date.today().isoformat()}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
