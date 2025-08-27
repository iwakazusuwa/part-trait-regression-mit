# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
# =========================
# ディレクトリ・ファイル設定
# =========================
INPUT_FOLDER = '2_data'
OUTPUT_FOLDER = '3_output'

parent_path = os.path.dirname(os.getcwd())
input_path = os.path.join(parent_path, INPUT_FOLDER, 'area_image.csv')
output_path = os.path.join(parent_path, OUTPUT_FOLDER)
os.makedirs(output_path, exist_ok=True)

save_name_1 = os.path.join(output_path, "1_分析条件を満たしたデータ.csv")
save_csv_radar = os.path.join(output_path, "2_回帰集計.csv")
save_csv_wide_all = os.path.join(output_path, "3_回帰結果_横長.csv")

save_pdf_he = os.path.join(output_path, "回帰結果_まとめ.pdf")
save_pdf_pl = os.path.join(output_path, "散布図と棒グラフまとめ.pdf")
save_pdf_re = os.path.join(output_path, "レーダーチャートまとめ.pdf")


# =========================
# 閾値
# =========================
threshold = 6

# =========================
# データ読み込み
# =========================
try:
    df = pd.read_csv(input_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding="cp932")

# ==========================================
# 車毎に image_/area_ を閾値でフィルタ
# ==========================================
def filter_columns_by_car(df, prefix, threshold):
    target_cols = [c for c in df.columns if c.startswith(prefix)]
    filtered_dfs = {}
    for car in df['Car'].unique():
        df_car = df[df['Car'] == car]
        counts = df_car[target_cols].apply(lambda col: col.isin([1, -1]).sum())
        keep_cols = counts[counts >= threshold].index.tolist()
        filtered_dfs[car] = df_car[['ID', 'Car'] + keep_cols]
    return filtered_dfs

df_image_filtered = filter_columns_by_car(df, "image_", threshold)
df_area_filtered = filter_columns_by_car(df, "area_", threshold)
df_filtered_by_car = {car: pd.merge(df_image_filtered[car], df_area_filtered[car], on=['ID', 'Car'])
                      for car in df['Car'].unique()}
df_all = pd.concat(df_filtered_by_car.values(), ignore_index=True)
df_all.to_csv(save_name_1, index=False, encoding="utf-8-sig")

# =========================
# 回帰分析
# =========================
trait_cols = [c for c in df_all.columns if c.startswith("image_")]
area_cols = [c for c in df_all.columns if c.startswith("area_")]
trait_cols_clean = [c.replace("image_", "") for c in trait_cols]
area_cols_clean = [c.replace("area_", "") for c in area_cols]

all_results, all_pvalues = [], []

for car in df_all['Car'].unique():
    df_car = df_all[df_all['Car'] == car].copy()
    trait_result = pd.DataFrame(index=trait_cols_clean, columns=area_cols_clean)
    trait_pvalues = pd.DataFrame(index=trait_cols_clean, columns=area_cols_clean)
    counts = {}

    for trait_orig, trait_clean in zip(trait_cols, trait_cols_clean):
        y = df_car[trait_orig].fillna(0)
        X = df_car[area_cols].fillna(0)
        counts[trait_clean] = (y != 0).sum()

        if y.nunique() <= 1:
            trait_result.loc[trait_clean] = [0] * len(area_cols_clean)
            trait_pvalues.loc[trait_clean] = [np.nan] * len(area_cols_clean)
            continue

        X_scaled = StandardScaler().fit_transform(X)
        X_sm = sm.add_constant(X_scaled)
        results_sm = sm.OLS(y, X_sm).fit()
        trait_result.loc[trait_clean] = np.round(results_sm.params[1:].values, 3)
        trait_pvalues.loc[trait_clean] = np.round(results_sm.pvalues[1:].values, 3)

    trait_result.insert(0, "Count", pd.Series(counts))
    trait_pvalues.insert(0, "Count", pd.Series(counts))
    trait_result = trait_result.sort_values(by="Count", ascending=False)
    trait_pvalues = trait_pvalues.loc[trait_result.index]
    trait_result["Car"] = car
    trait_pvalues["Car"] = car
    trait_result = trait_result.reset_index().rename(columns={"index": "Trait"})
    trait_pvalues = trait_pvalues.reset_index().rename(columns={"index": "Trait"})

    all_results.append(trait_result)
    all_pvalues.append(trait_pvalues)

final_result = pd.concat(all_results, ignore_index=True)
final_pvalues = pd.concat(all_pvalues, ignore_index=True)
final_result = final_result[final_result['Count'] > 0].reset_index(drop=True)
final_pvalues = final_pvalues[final_pvalues['Count'] > 0].reset_index(drop=True)

# =========================
# 回帰係数・P値・logP
# =========================
all_radar_data = []
columns_radar = area_cols_clean
cars = df_all['Car'].unique()
for car in cars:
    coefs = final_result[final_result['Car']==car].set_index('Trait')[columns_radar].astype(float)
    pvals = final_pvalues[final_pvalues['Car']==car].set_index('Trait')[columns_radar].apply(pd.to_numeric, errors='coerce')
    for trait in coefs.index:
        for part in columns_radar:
            all_radar_data.append({
                "Trait": trait,
                "Part": part,
                "Coef": coefs.loc[trait, part],
                "pvalue": pvals.loc[trait, part],
                "Car": car
            })

df_radar = pd.DataFrame(all_radar_data)
df_radar["logP"] = -np.log10(df_radar["pvalue"].clip(lower=1e-10))
df_radar.to_csv(save_csv_radar, index=False, encoding="utf-8-sig")

# %%
# =========================
# final_result から Car/Trait/Count を抽出
# =========================
df_count = final_result[['Car', 'Trait', 'Count']].drop_duplicates()

# df_radar を横長に変換
df_coef_wide = df_radar.pivot_table(
    index=["Car", "Trait"],
    columns="Part",
    values="Coef"
).reset_index()

df_logp_wide = df_radar.pivot_table(
    index=["Car", "Trait"],
    columns="Part",
    values="logP"
).reset_index()

# Count 列を正確にマージ
df_wide_all = pd.merge(df_coef_wide, df_logp_wide, on=["Car", "Trait"], suffixes=("_Coef", "_logP"))
df_wide_all = pd.merge(df_wide_all, df_count, on=["Car", "Trait"], how="left")

# 列順を整理
cols_order = ["Car", "Trait", "Count"] + [c for c in df_wide_all.columns if c not in ["Car", "Trait", "Count"]]
df_wide_all = df_wide_all[cols_order]

# CSV 保存
df_wide_all.to_csv(save_csv_wide_all, index=False, encoding="utf-8-sig")


# %%
# =========================
# 共通設定
# =========================
exclude_cols = ["Car", "Trait", "Count"]
columns_to_plot = [c for c in final_result.columns if c not in exclude_cols]
cars = final_result["Car"].unique()
n_cars = len(cars)


# ====================================
# 回帰係数・-log10(P値)ヒートマップPDF
# ====================================
fig, axes = plt.subplots(2, n_cars, figsize=(12*n_cars, 16), squeeze=False)
for idx, car in enumerate(cars):
    coefs = final_result[final_result['Car']==car].set_index('Trait')[columns_to_plot].astype(float)
    sns.heatmap(coefs.T, annot=True, annot_kws={"size": 20}, cmap="coolwarm", center=0,
                linewidths=0.5, cbar_kws={'label': '回帰係数'}, ax=axes[0, idx]) 
    axes[0, idx].set_xticklabels(axes[0, idx].get_xticklabels(), fontsize=16)
    axes[0, idx].set_yticklabels(axes[0, idx].get_yticklabels(), fontsize=16)
    axes[0, idx].set_title(f"{car}：回帰係数", fontsize=14)

    pvals = final_pvalues[final_pvalues['Car']==car].set_index('Trait')[columns_to_plot].apply(pd.to_numeric, errors='coerce')
    log_pvals = -np.log10(pvals.clip(lower=1e-10))
    sns.heatmap(log_pvals.T, annot=True, annot_kws={"size": 20}, cmap="YlGnBu",
                linewidths=0.5, cbar_kws={'label': '-log10(P値)'}, ax=axes[1, idx])    
    axes[1, idx].set_xticklabels(axes[1, idx].get_xticklabels(), fontsize=16)
    axes[1, idx].set_yticklabels(axes[1, idx].get_yticklabels(), fontsize=16)
    axes[1, idx].set_title(f"{car}：-log10(P値)", fontsize=14)

plt.tight_layout()
plt.savefig(save_pdf_he, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()


# %%
# =========================
# 散布図・棒グラフPDF
# =========================
def prepare_data(car):
    coefs = final_result[final_result['Car']==car].set_index('Trait')[columns_to_plot]
    pvals = final_pvalues[final_pvalues['Car']==car].set_index('Trait')[columns_to_plot].apply(pd.to_numeric, errors='coerce')
    log_pvals = -np.log10(pvals.clip(lower=1e-10))
    df_coef_melt = coefs.reset_index().melt(id_vars='Trait', var_name='Part', value_name='Coef')
    df_logp_melt = log_pvals.reset_index().melt(id_vars='Trait', var_name='Part', value_name='logP')
    df_merge = pd.merge(df_coef_melt, df_logp_melt, on=['Trait', 'Part'])
    df_merge['Label'] = df_merge['Trait'] + " × " + df_merge['Part']
    df_merge['Car'] = car
    return df_merge

all_data = [prepare_data(car) for car in cars]

with PdfPages(save_pdf_pl) as pdf:
    fig, axes = plt.subplots(2, n_cars, figsize=(8*n_cars, 12), squeeze=False)
    for idx, df_merge in enumerate(all_data):
        car = df_merge['Car'].iloc[0]
        # 散布図
        ax = axes[0, idx]
        # 正のCoefは赤、負のCoefは青、濃淡＝logP
        norm = mcolors.TwoSlopeNorm(vmin=df_merge['Coef'].min(),
                                    vcenter=0,
                                    vmax=df_merge['Coef'].max())
        scatter = ax.scatter(df_merge["Coef"], df_merge["logP"],
                             c=df_merge["Coef"], cmap="RdBu_r", norm=norm, s=80, edgecolor='k')
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel("回帰係数", fontsize=12)
        ax.set_ylabel("-log10(P値)", fontsize=12)
        ax.set_title(f"{car}：パーツの影響と信頼性", fontsize=14)
        for _, row in df_merge.iterrows():
            ax.text(row["Coef"], row["logP"], f'{row["Part"]}\n({row["Trait"]})',
                    fontsize=10, ha='right')

        # ---- 棒グラフ ----
        ax = axes[1, idx]
        # logP を 0-1 に正規化
        norm_logp = df_merge['logP'] / df_merge['logP'].max()        
        # Coef の符号ごとに色を割り当て
        colors_bar = [plt.cm.Reds(v) if c > 0 else plt.cm.Blues(v) for c, v in zip(df_merge['Coef'], norm_logp)]
        ax.barh(df_merge['Label'], df_merge['Coef'], color=colors_bar, edgecolor='black')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("回帰係数", fontsize=12)
        ax.set_title(f"{car}：パーツごとの回帰係数（色＝有意性）", fontsize=14)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# =========================
# レーダー描画関数
# =========================
def plot_radar_ax(ax, df, parts, angles, traits, title, yticks=None):
    for part in parts:
        vals = df.loc[part].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=part)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(v) for v in yticks], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=10)
    ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(title, fontsize=12)

# =========================
# レーダーチャートPDF
# =========================
with PdfPages(save_pdf_re) as pdf:
    fig, axes = plt.subplots(2, n_cars, figsize=(6*n_cars, 6*2), subplot_kw={'polar': True}, squeeze=False)
    for idx, car in enumerate(cars):
        coefs = final_result[final_result['Car']==car].set_index('Trait')[columns_radar].astype(float)
        traits = coefs.columns.tolist()
        parts = coefs.index.tolist()
        angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False).tolist()
        angles += angles[:1]
        ax = axes[0, idx]
        plot_radar_ax(ax, coefs, parts, angles, traits, f"{car}：回帰係数", yticks=[-0.2,-0.1,0,0.1,0.2])

        pvals = final_pvalues[final_pvalues['Car']==car].set_index('Trait')[columns_radar].apply(pd.to_numeric, errors='coerce')
        log_pvals = -np.log10(pvals.clip(lower=1e-10))
        ax = axes[1, idx]
        plot_radar_ax(ax, log_pvals, parts, angles, traits, f"{car}：-log10(P値)", yticks=[0,0.5,1,1.5,2])

    fig.legend(parts, loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=10)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# %%

# =============================================
# 出力フォルダを開く
# =============================================
if sys.platform.startswith('win'):
    os.startfile(output_path)
elif sys.platform.startswith('darwin'):
    subprocess.run(['open', output_path])
else:
    subprocess.run(['xdg-open', output_path])

print("完了")

# %%
