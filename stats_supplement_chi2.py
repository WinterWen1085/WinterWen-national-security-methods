# -*- coding: utf-8 -*-
"""
国家安全学研究方法论文：补充统计检验脚本
功能：
1. 基于 BN代码/scale_labels_bn_ready.csv 计算分类变量 χ² 检验、p值、Cramér's V；
2. 输出附录精简表、正文极简说明、低频单元格检查、主要标准化残差；
3. 保持 BN 与传统显著性检验的分工：χ²/p值负责基础关联检验，BN负责辅助结构分析。

运行位置：
请将本脚本放在仓库根目录 WinterWen-national-security-methods/ 下运行：
    python stats_supplement_chi2.py

依赖：
    pandas scipy numpy openpyxl
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "BN代码" / "scale_labels_bn_ready.csv",
    ROOT / "BN准备文件" / "BN-就绪总表（已合并仲裁补丁，离散化完成）.csv",
    ROOT / "附录部分" / "最终编码表.csv",
]
OUT_DIR = ROOT / "stats_supplement_outputs"
OUT_DIR.mkdir(exist_ok=True)

METHOD_TAG_LABELS = {
    10: "规范路径",
    20: "实证路径",
    30: "技术/模型路径",
    40: "思辨/理论路径",
}
METHOD_FAMILY_LABELS = {
    901: "统计/调查经验数据型",
    902: "实验/准实验/推断型",
    903: "技术建模与计算方法",
    904: "个案/比较/历史-过程追踪",
    905: "质性证据与解释",
    906: "文本/内容/话语分析",
    907: "政策分析与评估",
    908: "混合/综合方法",
    909: "规范/理论与构想",
}

def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到入模数据表。请确认以下任一路径存在：\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )

def year_bin(y: int | float | str) -> int:
    try:
        y = int(float(y))
    except Exception:
        return 0
    if 1999 <= y <= 2014:
        return 1
    if 2015 <= y <= 2019:
        return 2
    if 2020 <= y <= 2025:
        return 3
    return 0

def cramers_v_bias_corrected(table: pd.DataFrame) -> float:
    # Bergsma/Wicher bias-corrected Cramér's V
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return float("nan")
    return math.sqrt(phi2corr / denom)

def cramers_v_raw(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * (min(r - 1, k - 1))
    return math.sqrt(chi2 / denom) if denom > 0 else float("nan")

def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def std_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (obs - exp) / np.sqrt(exp)

def run_test(df: pd.DataFrame, name: str, row_var: str, col_var: str, note: str = "") -> dict:
    tab = pd.crosstab(df[row_var], df[col_var], dropna=False)
    chi2, p, dof, expected = chi2_contingency(tab, correction=False)
    expected = pd.DataFrame(expected, index=tab.index, columns=tab.columns)
    low_exp_count = int((expected < 5).sum().sum())
    low_exp_ratio = low_exp_count / expected.size if expected.size else 0

    res = std_residuals(tab.to_numpy(), expected.to_numpy())
    residual_rows = []
    for i, rv in enumerate(tab.index):
        for j, cv in enumerate(tab.columns):
            residual_rows.append({
                "检验": name,
                "行变量": row_var,
                "列变量": col_var,
                "行取值": rv,
                "列取值": cv,
                "观测频数": int(tab.iloc[i, j]),
                "期望频数": float(expected.iloc[i, j]),
                "标准化残差": float(res[i, j]),
            })
    residual_df = pd.DataFrame(residual_rows)
    top_resid = residual_df.reindex(residual_df["标准化残差"].abs().sort_values(ascending=False).index).head(8)

    return {
        "检验": name,
        "行变量": row_var,
        "列变量": col_var,
        "χ²": chi2,
        "df": dof,
        "p": p,
        "p_fmt": fmt_p(p),
        "Cramérs_V_原始": cramers_v_raw(tab),
        "Cramérs_V_校正": cramers_v_bias_corrected(tab),
        "样本量": int(tab.to_numpy().sum()),
        "低期望频数单元格数": low_exp_count,
        "低期望频数比例": low_exp_ratio,
        "解释提示": note,
        "crosstab": tab,
        "expected": expected,
        "top_residuals": top_resid,
    }

def collapse_low_freq_method_family(x):
    # 稳健性口径：将低频且解释相近的905/907并入“其他低频/应用评估类”
    try:
        x = int(x)
    except Exception:
        return x
    if x in (905, 907):
        return 990
    return x

def main():
    data_path = find_data()
    df = pd.read_csv(data_path)
    if "YearBin" not in df.columns:
        df["YearBin"] = df["year"].apply(year_bin)
    if "HighSensSecrecy" not in df.columns:
        df["HighSensSecrecy"] = ((df["PoliticalSensitivity"] == 3) & (df["SecrecyConstraint"] == 3)).astype(int)

    # 稳健性：低频门类合并口径
    df["MethodFamily_collapsed"] = df["MethodFamily"].apply(collapse_low_freq_method_family)

    tests = [
        ("T1 方法主轴的阶段差异", "YearBin", "MethodTag", "用于支撑方法演进判断"),
        ("T2 方法门类的阶段差异", "YearBin", "MethodFamily", "用于支撑方法门类演进判断；注意低频门类"),
        ("T2b 方法门类阶段差异（低频合并）", "YearBin", "MethodFamily_collapsed", "用于检查低频门类对T2的影响"),
        ("T3 H1：时间分箱与政策显著性", "YearBin", "PolicySalience", "用于检验阶段变化与政策议程显著性之间的列联关系"),
        ("T4 H2：政治敏感性与保密约束", "PoliticalSensitivity", "SecrecyConstraint", "用于检验高敏感与高保密是否存在关联"),
        ("T5a H3：数据可得性与方法主轴", "DataAccess", "MethodTag", "用于检验数据可得性与研究路径差异"),
        ("T5b H3：数据可得性与方法门类", "DataAccess", "MethodFamily", "用于检验数据可得性与具体方法配置差异；注意低频门类"),
        ("T5c H3：数据可得性与方法门类（低频合并）", "DataAccess", "MethodFamily_collapsed", "用于检查低频门类对T5b的影响"),
        ("T6a H4：高敏高密组合与方法主轴", "HighSensSecrecy", "MethodTag", "用于检验高敏高密情境下方法主轴是否收敛"),
        ("T6b H4：高敏高密组合与方法门类", "HighSensSecrecy", "MethodFamily", "用于检验高敏高密情境下方法门类是否收敛；注意低频门类"),
        ("T6c H4：高敏高密组合与方法门类（低频合并）", "HighSensSecrecy", "MethodFamily_collapsed", "用于检查低频门类对T6b的影响"),
    ]

    results = []
    crosstab_sheets = {}
    residuals_all = []
    expected_all = []

    for t in tests:
        r = run_test(df, *t)
        results.append({k: v for k, v in r.items() if k not in ("crosstab", "expected", "top_residuals")})
        safe_name = r["检验"][:25].replace("：", "_").replace("/", "_")
        crosstab_sheets[safe_name + "_obs"] = r["crosstab"]
        crosstab_sheets[safe_name + "_exp"] = r["expected"]
        residuals_all.append(r["top_residuals"])
        exp_long = r["expected"].stack().reset_index()
        exp_long.columns = [r["行变量"], r["列变量"], "期望频数"]
        exp_long.insert(0, "检验", r["检验"])
        expected_all.append(exp_long)

    summary = pd.DataFrame(results)
    residuals = pd.concat(residuals_all, ignore_index=True)
    expected_long = pd.concat(expected_all, ignore_index=True)

    # 数值格式保留
    for col in ["χ²", "Cramérs_V_原始", "Cramérs_V_校正", "低期望频数比例"]:
        summary[col] = summary[col].astype(float).round(4)
    summary["p"] = summary["p"].astype(float)

    xlsx_path = OUT_DIR / "补充统计检验结果_χ2_p值_CramersV.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="χ2汇总表", index=False)
        residuals.to_excel(writer, sheet_name="主要标准化残差", index=False)
        expected_long.to_excel(writer, sheet_name="期望频数检查", index=False)
        for name, tab in crosstab_sheets.items():
            # Excel sheet name max 31
            tab.to_excel(writer, sheet_name=name[:31])

    md_table = summary[["检验", "行变量", "列变量", "χ²", "df", "p_fmt", "Cramérs_V_校正", "低期望频数单元格数", "解释提示"]].copy()
    md_table.rename(columns={"p_fmt": "p", "Cramérs_V_校正": "Cramér's V（校正）"}, inplace=True)

    md_path = OUT_DIR / "附录用_主要变量关系χ2检验汇总表.md"
    md_path.write_text(md_table.to_markdown(index=False), encoding="utf-8")

    text = f"""正文极简说明建议：

为增强分类变量关系检验的统计基础，本文在封版入模表基础上补充χ²独立性检验，并同步报告p值与Cramér's V效应量。相关检验主要用于判断时间分箱、情境变量与方法变量之间是否存在统计关联；贝叶斯网络则仅承担辅助结构分析功能，用于观察上述关系在多变量条件依赖结构中是否呈现稳定连接。χ²检验及效应量结果见附录X，完整列联表、期望频数检查与标准化残差结果见开源仓库过程文件。

数据文件：{data_path.as_posix()}
输出文件：
- {xlsx_path.as_posix()}
- {md_path.as_posix()}
"""
    (OUT_DIR / "正文极简说明.txt").write_text(text, encoding="utf-8")

    print("完成。输出目录：", OUT_DIR)
    print(summary[["检验", "χ²", "df", "p_fmt", "Cramérs_V_校正", "低期望频数单元格数"]].to_string(index=False))

if __name__ == "__main__":
    main()
