
import streamlit as st
import pandas as pd
import numpy as np
import io
import math
import datetime as dt
from typing import Dict, Tuple
import openpyxl  # noqa: F401  # Ensure Excel engine is available

st.set_page_config(
    page_title="経営計画策定（単年）｜Streamlit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULTS = {
    "sales": 1000000000,
    "fte": 20.0,
    "cogs_mat_rate": 0.25,
    "cogs_lbr_rate": 0.06,
    "cogs_out_src_rate": 0.1,
    "cogs_out_con_rate": 0.04,
    "cogs_oth_rate": 0.0,
    "opex_h_rate": 0.17,
    "opex_k_rate": 0.468,
    "opex_dep_rate": 0.006,
    "noi_misc_rate": 0.0001,
    "noi_grant_rate": 0.0,
    "noi_oth_rate": 0.0,
    "noe_int_rate": 0.0074,
    "noe_oth_rate": 0.0,
    "unit": "百万円",
    "fiscal_year": 2025
}

ITEMS = [
    ("REV", "売上高", "売上"),
    ("COGS_MAT", "外部仕入｜材料費", "外部仕入"),
    ("COGS_LBR", "外部仕入｜労務費(外部)", "外部仕入"),
    ("COGS_OUT_SRC", "外部仕入｜外注費(専属)", "外部仕入"),
    ("COGS_OUT_CON", "外部仕入｜外注費(委託)", "外部仕入"),
    ("COGS_OTH", "外部仕入｜その他諸経費", "外部仕入"),
    ("COGS_TTL", "外部仕入｜計", "外部仕入"),
    ("GROSS", "粗利(加工高)", "粗利"),
    ("OPEX_H", "内部費用｜人件費", "内部費用"),
    ("OPEX_K", "内部費用｜経費", "内部費用"),
    ("OPEX_DEP", "内部費用｜減価償却費", "内部費用"),
    ("OPEX_TTL", "内部費用｜計", "内部費用"),
    ("OP", "営業利益", "損益"),
    ("NOI_MISC", "営業外収益｜雑収入", "営業外"),
    ("NOI_GRANT", "営業外収益｜補助金/給付金", "営業外"),
    ("NOI_OTH", "営業外収益｜その他", "営業外"),
    ("NOE_INT", "営業外費用｜支払利息", "営業外"),
    ("NOE_OTH", "営業外費用｜雑損", "営業外"),
    ("ORD", "経常利益", "損益"),
    ("BE_SALES", "損益分岐点売上高", "KPI"),
    ("PC_SALES", "一人当たり売上", "KPI"),
    ("PC_GROSS", "一人当たり粗利", "KPI"),
    ("PC_ORD", "一人当たり経常利益", "KPI"),
    ("LDR", "労働分配率", "KPI")
]

# Mapping from item code to label for quick lookup
ITEM_LABELS = {code: label for code, label, _ in ITEMS}

def millions(x):
    return x / 1_000_000

def thousands(x):
    return x / 1_000

def format_money(x, unit="百万円"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if unit == "百万円":
        return f"{millions(x):,.1f}"
    elif unit == "千円":
        return f"{thousands(x):,.0f}"
    else:
        return f"{x:,.0f}"

class PlanConfig:
    def __init__(self, base_sales, fte, unit):
        self.base_sales = base_sales
        self.fte = max(0.0001, fte)
        self.unit = unit
        self.items = {}

    def set_rate(self, code, rate, rate_base='sales'):
        self.items[code] = {"method": "rate", "value": float(rate), "rate_base": rate_base}

    def set_amount(self, code, amount):
        self.items[code] = {"method": "amount", "value": float(amount), "rate_base": "fixed"}

    def clone(self):
        c = PlanConfig(self.base_sales, self.fte, self.unit)
        c.items = {k: v.copy() for k, v in self.items.items()}
        return c

def compute(plan: PlanConfig, sales_override=None, amount_overrides: Dict[str, float]=None) -> Dict[str, float]:
    S = float(plan.base_sales if sales_override is None else sales_override)
    amt = {code: 0.0 for code, *_ in ITEMS}
    amt["REV"] = S

    def line_amount(code, gross_guess):
        cfg = plan.items.get(code, None)
        if amount_overrides and code in amount_overrides:
            return float(amount_overrides[code])
        if cfg is None:
            return 0.0
        if cfg["method"] == "amount":
            return float(cfg["value"])
        r = float(cfg["value"])
        base = cfg.get("rate_base", "sales")
        if base == "sales":
            return S * r
        elif base == "gross":
            return max(0.0, gross_guess) * r
        elif base == "fixed":
            return r
        return S * r

    cogs_codes = ["COGS_MAT", "COGS_LBR", "COGS_OUT_SRC", "COGS_OUT_CON", "COGS_OTH"]
    sales_based_cogs = 0.0
    for code in cogs_codes:
        cfg = plan.items.get(code)
        if cfg and cfg["method"] == "rate" and cfg.get("rate_base", "sales") == "sales":
            sales_based_cogs += S * float(cfg["value"])
        elif cfg and cfg["method"] == "amount":
            sales_based_cogs += float(cfg["value"])

    gross = S - sales_based_cogs
    for _ in range(5):
        cogs = 0.0
        for code in cogs_codes:
            cogs += max(0.0, line_amount(code, gross))
        gross_new = S - cogs
        if abs(gross_new - gross) < 1e-6:
            gross = gross_new
            break
        gross = gross_new

    cogs_total = 0.0
    for code in cogs_codes:
        val = max(0.0, line_amount(code, gross))
        amt[code] = val
        cogs_total += val
    amt["COGS_TTL"] = cogs_total
    amt["GROSS"] = S - cogs_total

    opex_codes = ["OPEX_H", "OPEX_K", "OPEX_DEP"]
    opex_total = 0.0
    for code in opex_codes:
        val = max(0.0, line_amount(code, amt["GROSS"]))
        amt[code] = val
        opex_total += val
    amt["OPEX_TTL"] = opex_total

    amt["OP"] = amt["GROSS"] - amt["OPEX_TTL"]

    noi_codes = ["NOI_MISC", "NOI_GRANT", "NOI_OTH"]
    noe_codes = ["NOE_INT", "NOE_OTH"]
    for code in noi_codes + noe_codes:
        val = max(0.0, line_amount(code, amt["GROSS"]))
        amt[code] = val

    amt["ORD"] = amt["OP"] + (amt["NOI_MISC"] + amt["NOI_GRANT"] + amt["NOI_OTH"]) - (amt["NOE_INT"] + amt["NOE_OTH"])

    var_cost = 0.0
    for code in cogs_codes + opex_codes + noi_codes + noe_codes:
        cfg = plan.items.get(code)
        if cfg and cfg["method"] == "rate" and cfg.get("rate_base", "sales") in ("sales", "gross"):
            if cfg.get("rate_base") == "gross":
                g_ratio = amt["GROSS"] / S if S > 0 else 0.0
                var_cost += S * (cfg["value"] * g_ratio)
            else:
                var_cost += S * cfg["value"]

    fixed_cost = 0.0
    for code in cogs_codes + opex_codes + noi_codes + noe_codes:
        cfg = plan.items.get(code)
        if cfg and cfg["method"] == "amount":
            fixed_cost += cfg["value"]
        elif cfg and cfg.get("rate_base") == "fixed":
            if cfg["method"] == "rate":
                fixed_cost += cfg["value"]
    cm_ratio = 1.0 - (var_cost / S if S > 0 else 0.0)
    if cm_ratio <= 0:
        be_sales = float("inf")
    else:
        be_sales = fixed_cost / cm_ratio
    amt["BE_SALES"] = be_sales

    fte = max(0.0001, plan.fte)
    amt["PC_SALES"] = amt["REV"] / fte
    amt["PC_GROSS"] = amt["GROSS"] / fte
    amt["PC_ORD"] = amt["ORD"] / fte
    amt["LDR"] = (amt["OPEX_H"] / amt["GROSS"]) if amt["GROSS"] > 0 else np.nan

    return amt

def bisection_for_target_op(plan: PlanConfig, target_op: float, s_low: float, s_high: float, max_iter=60, eps=1_000.0) -> Tuple[float, Dict[str, float]]:
    def op_at(S):
        return compute(plan, sales_override=S)["ORD"]
    low, high = max(0.0, s_low), max(s_low * 1.5, s_high)
    f_low = op_at(low)
    f_high = op_at(high)
    it = 0
    while (f_low - target_op) * (f_high - target_op) > 0 and high < 1e13 and it < 40:
        high = high * 1.6 if high > 0 else 1_000_000.0
        f_high = op_at(high)
        it += 1
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = op_at(mid)
        if abs(f_mid - target_op) <= eps:
            return mid, compute(plan, sales_override=mid)
        if (f_low - target_op) * (f_mid - target_op) <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    mid = 0.5 * (low + high)
    return mid, compute(plan, sales_override=mid)

# Sidebar
with st.sidebar:
    st.header("⚙️ 基本設定")
    fiscal_year = st.number_input("会計年度", value=int(DEFAULTS["fiscal_year"]), step=1, format="%d")
    unit = st.selectbox("表示単位", ["百万円", "千円", "円"], index=0, help="計算は円ベース、表示のみ丸めます。")
    base_sales = st.number_input("売上高（ベース）", value=float(DEFAULTS["sales"]), step=10_000_000.0, min_value=0.0, format="%.0f")
    fte = st.number_input("人員数（FTE換算）", value=float(DEFAULTS["fte"]), step=1.0, min_value=0.0)

    st.markdown("---")
    st.caption("外部仕入（売上対・初期値）")
    cogs_mat_r = st.number_input("材料費 率", value=float(DEFAULTS["cogs_mat_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    cogs_lbr_r = st.number_input("労務費(外部) 率", value=float(DEFAULTS["cogs_lbr_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    cogs_out_src_r = st.number_input("外注費(専属) 率", value=float(DEFAULTS["cogs_out_src_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    cogs_out_con_r = st.number_input("外注費(委託) 率", value=float(DEFAULTS["cogs_out_con_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    cogs_oth_r = st.number_input("その他諸経費 率", value=float(DEFAULTS["cogs_oth_rate"]), step=0.005, min_value=0.0, max_value=3.0, format="%.3f")

    st.markdown("---")
    st.caption("内部費用（売上対・初期値）")
    opex_h_r = st.number_input("人件費 率", value=float(DEFAULTS["opex_h_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    opex_k_r = st.number_input("経費 率", value=float(DEFAULTS["opex_k_rate"]), step=0.01, min_value=0.0, max_value=3.0, format="%.3f")
    opex_dep_r = st.number_input("減価償却 率", value=float(DEFAULTS["opex_dep_rate"]), step=0.001, min_value=0.0, max_value=3.0, format="%.3f")

    st.markdown("---")
    st.caption("営業外（売上対・初期値）")
    noi_misc_r = st.number_input("営業外収益：雑収入 率", value=float(DEFAULTS["noi_misc_rate"]), step=0.0005, min_value=0.0, max_value=1.0, format="%.4f")
    noi_grant_r = st.number_input("営業外収益：補助金 率", value=float(DEFAULTS["noi_grant_rate"]), step=0.0005, min_value=0.0, max_value=1.0, format="%.4f")
    noi_oth_r = st.number_input("営業外収益：その他 率", value=float(DEFAULTS["noi_oth_rate"]), step=0.0005, min_value=0.0, max_value=1.0, format="%.4f")
    noe_int_r = st.number_input("営業外費用：支払利息 率", value=float(DEFAULTS["noe_int_rate"]), step=0.0005, min_value=0.0, max_value=1.0, format="%.4f")
    noe_oth_r = st.number_input("営業外費用：雑損 率", value=float(DEFAULTS["noe_oth_rate"]), step=0.0005, min_value=0.0, max_value=1.0, format="%.4f")

base_plan = PlanConfig(base_sales=base_sales, fte=fte, unit=unit)
base_plan.set_rate("COGS_MAT", cogs_mat_r, "sales")
base_plan.set_rate("COGS_LBR", cogs_lbr_r, "sales")
base_plan.set_rate("COGS_OUT_SRC", cogs_out_src_r, "sales")
base_plan.set_rate("COGS_OUT_CON", cogs_out_con_r, "sales")
base_plan.set_rate("COGS_OTH", cogs_oth_r, "sales")

base_plan.set_rate("OPEX_H", opex_h_r, "sales")
base_plan.set_rate("OPEX_K", opex_k_r, "sales")
base_plan.set_rate("OPEX_DEP", opex_dep_r, "sales")

base_plan.set_rate("NOI_MISC", noi_misc_r, "sales")
base_plan.set_rate("NOI_GRANT", noi_grant_r, "sales")
base_plan.set_rate("NOI_OTH", noi_oth_r, "sales")
base_plan.set_rate("NOE_INT", noe_int_r, "sales")
base_plan.set_rate("NOE_OTH", noe_oth_r, "sales")

tab_input, tab_scen, tab_analysis, tab_export = st.tabs(["📝 計画入力", "🧪 シナリオ", "📊 感応度分析", "📤 エクスポート"])

with tab_input:
    st.subheader("単年利益計画（目標列）")
    base_amt = compute(base_plan)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("売上高", f"{format_money(base_amt['REV'], base_plan.unit)} {base_plan.unit}")
    c2.metric("粗利(加工高)", f"{format_money(base_amt['GROSS'], base_plan.unit)} {base_plan.unit}")
    c3.metric("営業利益", f"{format_money(base_amt['OP'], base_plan.unit)} {base_plan.unit}")
    c4.metric("経常利益", f"{format_money(base_amt['ORD'], base_plan.unit)} {base_plan.unit}")
    be_label = "∞" if not math.isfinite(base_amt["BE_SALES"]) else f"{format_money(base_amt['BE_SALES'], base_plan.unit)} {base_plan.unit}"
    c5.metric("損益分岐点売上高", be_label)

    c6, c7, c8 = st.columns(3)
    c6.metric("一人当たり売上", f"{format_money(base_amt['PC_SALES'], base_plan.unit)} {base_plan.unit}")
    c7.metric("一人当たり粗利", f"{format_money(base_amt['PC_GROSS'], base_plan.unit)} {base_plan.unit}")
    ldr = base_amt["LDR"]
    ldr_str = "—" if (ldr is None or not math.isfinite(ldr)) else f"{ldr*100:.1f}%"
    c8.metric("労働分配率", ldr_str)

    rows = []
    for code, label, group in ITEMS:
        if code in ("PC_SALES", "PC_GROSS", "PC_ORD", "LDR", "BE_SALES"):
            continue
        val = base_amt[code]
        rows.append({"項目": label, "金額": format_money(val, base_plan.unit)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=min(520, 40 + 28*len(rows)))

    st.info("ヒント: サイドバーの率・人員・売上を変えると、即座に計算結果が更新されます。金額入力を使いたい場合は、下の『金額上書き』を利用してください。")

    with st.expander("🔧 金額上書き（固定費/個別額の設定）", expanded=False):
        st.caption("金額が入力された項目は、率の指定より優先され固定費扱いになります。")
        col1, col2, col3 = st.columns(3)
        override_inputs = {}
        for i, code in enumerate(["COGS_MAT","COGS_LBR","COGS_OUT_SRC","COGS_OUT_CON","COGS_OTH","OPEX_H","OPEX_K","OPEX_DEP","NOI_MISC","NOI_GRANT","NOI_OTH","NOE_INT","NOE_OTH"]):
            if i % 3 == 0:
                c = col1
            elif i % 3 == 1:
                c = col2
            else:
                c = col3
            # Look up label without reconstructing the dictionary each time
            val = c.number_input(
                f"{ITEM_LABELS[code]}（金額上書き）",
                min_value=0.0,
                value=0.0,
                step=1_000_000.0,
                key=f"ov_{code}"
            )
            if val > 0:
                override_inputs[code] = val

        if st.button("上書きを反映", type="primary"):
            preview_amt = compute(base_plan, amount_overrides=override_inputs)
            st.session_state["overrides"] = override_inputs
            st.success("上書きを反映しました（この状態でシナリオにも適用されます）。")

            rows2 = []
            for code, label, group in ITEMS:
                if code in ("PC_SALES","PC_GROSS","PC_ORD","LDR","BE_SALES"):
                    continue
                before = base_amt[code]
                after = preview_amt[code]
                rows2.append({"項目": label, "前": format_money(before, base_plan.unit), "後": format_money(after, base_plan.unit)})
            st.dataframe(pd.DataFrame(rows2), use_container_width=True)

def scenario_table(plan: PlanConfig, unit, overrides):
    scenarios = [
        ("目標", {"type": "none"}),
        ("売上高10%増", {"type": "sales_pct", "value": 0.10}),
        ("売上高5%減", {"type": "sales_pct", "value": -0.05}),
        ("売上高10%減", {"type": "sales_pct", "value": -0.10}),
        ("粗利1%減", {"type": "gross_pt", "value": -0.01}),
        ("経常利益5千万円", {"type": "target_op", "value": 50_000_000.0}),
        ("昨年同一", {"type": "last_year"}),
        ("損益分岐点売上高", {"type": "bep"}),
    ]
    st.caption("各シナリオのラベルとパラメータを編集できます。")
    editable = []
    for i, (name, spec) in enumerate(scenarios):
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        new_name = col1.text_input(f"シナリオ{i+1} 名称", value=name, key=f"scn_name_{i}")
        typ = col2.selectbox("タイプ", options=["none","sales_pct","gross_pt","target_op","last_year","bep"], index=["none","sales_pct","gross_pt","target_op","last_year","bep"].index(spec["type"]), key=f"scn_type_{i}")
        if typ in ("sales_pct", "gross_pt"):
            val = col3.number_input("値（% / pt）", value=float(spec.get("value", 0.0)), step=0.01, format="%.3f", key=f"scn_val_{i}")
        elif typ == "target_op":
            val = col3.number_input("目標経常利益（円）", value=float(spec.get("value", 0.0)), step=1_000_000.0, format="%.0f", key=f"scn_val_{i}")
        else:
            val = None
            col3.write("—")
        editable.append((new_name, {"type": typ, "value": val}))

    st.markdown("---")
    cols = ["項目"] + [nm for nm, _ in editable]
    rows = {code: [label] for code, label, _ in ITEMS if code not in ("PC_SALES","PC_GROSS","PC_ORD","LDR","BE_SALES")}
    kpis = {"BE_SALES": ["損益分岐点売上高"], "PC_SALES": ["一人当たり売上"], "PC_GROSS": ["一人当たり粗利"], "PC_ORD": ["一人当たり経常利益"], "LDR": ["労働分配率"]}

    def apply_driver(spec, base_amt):
        t = spec["type"]
        v = spec.get("value", None)
        if t == "none":
            return plan.base_sales, overrides, None
        if t == "sales_pct":
            S = plan.base_sales * (1.0 + float(v))
            return S, overrides, None
        if t == "gross_pt":
            delta = float(v)
            S = plan.base_sales
            delta_e = -delta * S
            ov = dict(overrides) if overrides else {}
            current = ov.get("COGS_OTH", None)
            if current is None:
                tmp = compute(plan, sales_override=S, amount_overrides=ov)
                base_oth = tmp["COGS_OTH"]
                ov["COGS_OTH"] = max(0.0, base_oth + delta_e)
            else:
                ov["COGS_OTH"] = max(0.0, current + delta_e)
            return S, ov, None
        if t == "target_op":
            target = float(v)
            sol_S, sol_amt = bisection_for_target_op(plan, target, s_low=0.0, s_high=max(1.2*plan.base_sales, 1_000_000.0))
            return sol_S, overrides, sol_amt
        if t == "last_year":
            return plan.base_sales, overrides, None
        if t == "bep":
            temp = compute(plan, sales_override=plan.base_sales, amount_overrides=overrides)
            be = temp["BE_SALES"]
            return be if math.isfinite(be) else plan.base_sales, overrides, None
        return plan.base_sales, overrides, None

    base_amt = compute(plan, amount_overrides=overrides)
    for code, label, _ in ITEMS:
        if code in rows:
            rows[code].append(format_money(base_amt.get(code, 0.0), unit))
    for k in kpis.keys():
        kpis[k].append(format_money(base_amt.get(k, 0.0), unit) if k != "LDR" else (f"{base_amt.get('LDR', float('nan'))*100:.1f}%" if base_amt.get("LDR", float('nan'))==base_amt.get("LDR", float('nan')) else "—"))

    for i, (nm, spec) in enumerate(editable[1:], start=1):
        S_override, ov, pre_amt = apply_driver(spec, base_amt)
        scn_amt = compute(plan, sales_override=S_override, amount_overrides=ov) if pre_amt is None else pre_amt
        for code, label, _ in ITEMS:
            if code in rows:
                rows[code].append(format_money(scn_amt.get(code, 0.0), unit))
        for k in kpis.keys():
            if k == "LDR":
                v = scn_amt.get("LDR", float("nan"))
                kpis[k].append(f"{v*100:.1f}%" if v==v else "—")
            else:
                kpis[k].append(format_money(scn_amt.get(k, 0.0), unit))

    df1 = pd.DataFrame(rows.values(), columns=cols, index=rows.keys())
    df2 = pd.DataFrame(kpis.values(), columns=cols, index=kpis.keys())
    st.subheader("シナリオ比較（金額）")
    st.dataframe(df1, use_container_width=True)
    st.subheader("KPI（損益分岐点・一人当たり・労働分配率）")
    st.dataframe(df2, use_container_width=True)
    return df1, df2, editable

with tab_scen:
    overrides = st.session_state.get("overrides", {})
    df_amounts, df_kpis, scenario_specs = scenario_table(base_plan, unit, overrides)

with tab_analysis:
    st.subheader("感応度分析｜経常利益（ORD）への影響")
    st.caption("主要ドライバを±の変化で同時に比較（トルネード図）。スライダーで変化幅を指定してください。")
    c1, c2, c3, c4 = st.columns(4)
    pct_sales = c1.slider("売上高 変化率（±%）", min_value=0.0, max_value=50.0, value=10.0, step=1.0) / 100.0
    pt_gross = c2.slider("粗利率 変化（±pt）", min_value=0.0, max_value=10.0, value=1.0, step=0.5) / 100.0
    pct_personnel = c3.slider("人件費 変化率（±%）", min_value=0.0, max_value=50.0, value=10.0, step=1.0) / 100.0
    pct_expense = c4.slider("経費 変化率（±%）", min_value=0.0, max_value=50.0, value=10.0, step=1.0) / 100.0

    base_amt = compute(base_plan, amount_overrides=st.session_state.get("overrides", {}))
    base_op = base_amt["ORD"]

    def op_with_changes(ds=0.0, dgross_pt=0.0, dH=0.0, dK=0.0):
        plan = base_plan.clone()
        S = plan.base_sales * (1.0 + ds)

        overrides = st.session_state.get("overrides", {}).copy()
        if abs(dgross_pt) > 0:
            delta_e = -dgross_pt * S
            overrides["COGS_OTH"] = max(0.0, compute(plan, sales_override=S, amount_overrides=overrides)["COGS_OTH"] + delta_e)

        if "OPEX_H" in overrides and overrides["OPEX_H"] > 0:
            overrides["OPEX_H"] = overrides["OPEX_H"] * (1.0 + dH)
        else:
            val = compute(plan, sales_override=S, amount_overrides=overrides)["OPEX_H"]
            overrides["OPEX_H"] = max(0.0, val * (1.0 + dH))
        if "OPEX_K" in overrides and overrides["OPEX_K"] > 0:
            overrides["OPEX_K"] = overrides["OPEX_K"] * (1.0 + dK)
        else:
            val = compute(plan, sales_override=S, amount_overrides=overrides)["OPEX_K"]
            overrides["OPEX_K"] = max(0.0, val * (1.0 + dK))

        return compute(plan, sales_override=S, amount_overrides=overrides)["ORD"]

    changes = [
        ("売上高 +", op_with_changes(ds=+pct_sales) - base_op),
        ("売上高 -", op_with_changes(ds=-pct_sales) - base_op),
        ("粗利率 +", op_with_changes(dgross_pt=+pt_gross) - base_op),
        ("粗利率 -", op_with_changes(dgross_pt=-pt_gross) - base_op),
        ("人件費 +", op_with_changes(dH=+pct_personnel) - base_op),
        ("人件費 -", op_with_changes(dH=-pct_personnel) - base_op),
        ("経費 +", op_with_changes(dK=+pct_expense) - base_op),
        ("経費 -", op_with_changes(dK=-pct_expense) - base_op),
    ]
    changes_sorted = sorted(changes, key=lambda x: abs(x[1]), reverse=True)

    df_chg = pd.DataFrame({
        "ドライバ": [k for k, _ in changes_sorted],
        "OP変化（円）": [v for _, v in changes_sorted]
    })
    st.dataframe(df_chg, use_container_width=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    y_pos = np.arange(len(changes_sorted))
    values = [v for _, v in changes_sorted]
    labels = [k for k, _ in changes_sorted]
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("OP変化（円）")
    ax.set_title("トルネード図：経常利益への影響")
    st.pyplot(fig)

with tab_export:
    st.subheader("エクスポート")
    st.caption("ワンクリックでExcel出力（シート: 金額, KPI, 感応度）。PDFはExcelから印刷設定で作成してください。")

    def get_scenario_specs_from_session():
        specs = []
        defaults = ["目標","売上高10%増","売上高5%減","売上高10%減","粗利1%減","経常利益5千万円","昨年同一","損益分岐点売上高"]
        for i in range(8):
            nm = st.session_state.get(f"scn_name_{i}", defaults[i] if i < len(defaults) else f"シナリオ{i+1}")
            typ = st.session_state.get(f"scn_type_{i}", "none")
            if typ in ("sales_pct","gross_pt","target_op"):
                val = st.session_state.get(f"scn_val_{i}", 0.0)
            else:
                val = None
            specs.append((nm, {"type": typ, "value": val}))
        return specs

    specs = get_scenario_specs_from_session()

    def compute_scenario_numeric(plan, specs, overrides):
        cols = ["項目"] + [nm for nm,_ in specs]
        num_rows = {code: [label] for code, label, _ in ITEMS if code not in ("PC_SALES","PC_GROSS","PC_ORD","LDR","BE_SALES")}
        num_kpis = {"BE_SALES": ["損益分岐点売上高"], "PC_SALES": ["一人当たり売上"], "PC_GROSS": ["一人当たり粗利"], "PC_ORD": ["一人当たり経常利益"], "LDR": ["労働分配率"]}
        def apply_driver(spec):
            t = spec["type"]; v = spec.get("value", None)
            if t == "none": return plan.base_sales, overrides, None
            if t == "sales_pct": return plan.base_sales * (1.0 + float(v)), overrides, None
            if t == "gross_pt":
                S = plan.base_sales
                delta_e = -float(v) * S
                ov = dict(overrides) if overrides else {}
                tmp = compute(plan, sales_override=S, amount_overrides=ov)
                base_oth = tmp["COGS_OTH"]
                ov["COGS_OTH"] = max(0.0, base_oth + delta_e)
                return S, ov, None
            if t == "target_op":
                target = float(v)
                sol_S, sol_amt = bisection_for_target_op(plan, target, s_low=0.0, s_high=max(1.2*plan.base_sales, 1_000_000.0))
                return sol_S, overrides, sol_amt
            if t == "last_year":
                return plan.base_sales, overrides, None
            if t == "bep":
                temp = compute(plan, sales_override=plan.base_sales, amount_overrides=overrides)
                be = temp["BE_SALES"]
                return be if math.isfinite(be) else plan.base_sales, overrides, None
            return plan.base_sales, overrides, None

        base_amt = compute(plan, amount_overrides=overrides)
        for code, label, _ in ITEMS:
            if code in num_rows:
                num_rows[code].append(base_amt.get(code, 0.0))
        for k in num_kpis.keys():
            num_kpis[k].append(base_amt.get(k, 0.0))

        for (nm, spec) in specs[1:]:
            S, ov, pre = apply_driver(spec)
            scn_amt = compute(plan, sales_override=S, amount_overrides=ov) if pre is None else pre
            for code, label, _ in ITEMS:
                if code in num_rows:
                    num_rows[code].append(scn_amt.get(code, 0.0))
            for k in num_kpis.keys():
                num_kpis[k].append(scn_amt.get(k, 0.0))

        df_num = pd.DataFrame(num_rows.values(), columns=cols, index=num_rows.keys())
        df_kpi = pd.DataFrame(num_kpis.values(), columns=cols, index=num_kpis.keys())
        return df_num, df_kpi

    df_num, df_kpi = compute_scenario_numeric(base_plan, specs, st.session_state.get("overrides", {}))

    def recompute_sensitivity_table():
        base_amt = compute(base_plan, amount_overrides=st.session_state.get("overrides", {}))
        base_op = base_amt["ORD"]
        def op_with(ds=0.1, dgp=0.01, dH=0.1, dK=0.1):
            plan = base_plan.clone()
            S = plan.base_sales * (1.0 + ds)
            overrides = st.session_state.get("overrides", {}).copy()
            delta_e = -dgp * S
            overrides["COGS_OTH"] = max(0.0, compute(plan, sales_override=S, amount_overrides=overrides)["COGS_OTH"] + delta_e)
            val = compute(plan, sales_override=S, amount_overrides=overrides)["OPEX_H"]
            overrides["OPEX_H"] = max(0.0, val * (1.0 + dH))
            val = compute(plan, sales_override=S, amount_overrides=overrides)["OPEX_K"]
            overrides["OPEX_K"] = max(0.0, val * (1.0 + dK))
            return compute(plan, sales_override=S, amount_overrides=overrides)["ORD"]
        changes = [
            ("売上高 +10%", op_with(ds=+0.10) - base_op),
            ("売上高 -10%", op_with(ds=-0.10) - base_op),
            ("粗利率 +1pt", op_with(dgp=+0.01) - base_op),
            ("粗利率 -1pt", op_with(dgp=-0.01) - base_op),
            ("人件費 +10%", op_with(dH=+0.10) - base_op),
            ("人件費 -10%", op_with(dH=-0.10) - base_op),
            ("経費 +10%", op_with(dK=+0.10) - base_op),
            ("経費 -10%", op_with(dK=-0.10) - base_op),
        ]
        df = pd.DataFrame(changes, columns=["ドライバ","OP変化（円）"])
        return df

    df_sens = recompute_sensitivity_table()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_num.to_excel(writer, sheet_name="金額", index=True)
        df_kpi.to_excel(writer, sheet_name="KPI", index=True)
        df_sens.to_excel(writer, sheet_name="感応度", index=False)
    data = output.getvalue()

    st.download_button(
        label="📥 Excel（.xlsx）をダウンロード",
        data=data,
        file_name=f"利益計画_{dt.date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("© 経営計画策定WEBアプリ（Streamlit版） | 表示単位と計算単位を分離し、丸めの影響を最小化しています。")
