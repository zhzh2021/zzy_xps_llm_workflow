"""
Structure–Condition Plot Suite for XAS ML

Generates plots described in STRUCTURE_CONDITION_INTERPRETATION_GUIDE.md:
- anova_results.csv + anova_visualization.png
- feature_structure_linkage.png
- feature_structure_condition_linkage.csv
- condition_structure_impact.png
- condition_impact_analysis.png + condition_impact_ranking.csv
- feature_metadata_correlations.png (top correlations)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import f_oneway
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


FEATURE_STRUCTURE_MAP = {
    'e0': 'Fe Oxidation State',
    'edge_slope': 'Coordination Order/Disorder',
    'white_line_intensity': 'Fe-Ligand Covalency',
    'pre_edge_area': 'Coordination Geometry',
    'xanes_centroid': 'Average Electronic State',
    'post_edge_slope': 'Extended Coordination',
    'second_derivative_zero': 'Edge Shape/Curvature',
    'edge_step': 'Fe Concentration',
    'white_line_energy': 'Ligand Field Strength',
    'white_line_fwhm': 'Site Heterogeneity'
}


def _extract_group_labels(metadata_dict: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    """Map metadata to categorical groups used in ANOVA and impact plots."""
    anion = metadata_dict.get('iron_source', [])
    ligand = metadata_dict.get('ligand', [])

    anion_types = []
    ligand_types = []
    for a, l in zip(anion, ligand):
        if isinstance(a, str) and 'FeCl2' in a:
            anion_types.append('FeCl2')
        elif isinstance(a, str) and 'FeSO4' in a:
            anion_types.append('FeSO4')
        else:
            anion_types.append('Unknown')

        if isinstance(l, str) and 'Malic' in l:
            ligand_types.append('Malic')
        elif isinstance(l, str) and 'Tartaric' in l:
            ligand_types.append('Tartaric')
        else:
            ligand_types.append('Unknown')

    combined = [f"{a}+{l}" for a, l in zip(anion_types, ligand_types)]
    return {
        'anion_types': anion_types,
        'ligand_types': ligand_types,
        'combined_groups': combined
    }


def generate_structure_condition_plots(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    metadata_dict: Dict[str, List[Any]],
    trend_results,
    output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_matrix is None or feature_matrix.size == 0:
        return

    # 1) ANOVA across combined chemical groups
    if HAS_SCIPY:
        groups = _extract_group_labels(metadata_dict)
        combined_groups = groups['combined_groups']
        unique_groups = sorted(set(combined_groups))

        anova_results = []
        for feat_idx, feat_name in enumerate(feature_names):
            feat_values = feature_matrix[:, feat_idx]
            if not np.any(np.isfinite(feat_values)):
                continue

            group_data = []
            for group in unique_groups:
                vals = [feat_values[i] for i, g in enumerate(combined_groups) if g == group]
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) == 0:
                    vals = [np.nan]
                group_data.append(vals)

            try:
                f_stat, p_val = f_oneway(*group_data)
            except Exception:
                continue

            grand_mean = np.nanmean(feat_values)
            ss_between = sum([len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in group_data])
            ss_total = np.nansum((feat_values - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            anova_results.append({
                'Feature': feat_name,
                'F_statistic': round(float(f_stat), 4),
                'p_value': float(p_val),
                'Significant': 'Yes' if p_val < 0.05 else 'No',
                'eta_squared': round(float(eta_squared), 4),
                'Effect_size': 'Large' if eta_squared > 0.14 else ('Medium' if eta_squared > 0.06 else 'Small')
            })

        if anova_results:
            anova_df = pd.DataFrame(anova_results).sort_values('p_value')
            anova_df.to_csv(output_dir / "anova_results.csv", index=False)

            sig_features = anova_df[anova_df['Significant'] == 'Yes'].head(10)
            if not sig_features.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                ax1.barh(sig_features['Feature'], sig_features['F_statistic'],
                         color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2, height=0.5)
                ax1.set_xlabel('F-statistic', fontsize=16, fontweight='bold')
                ax1.set_ylabel('Feature', fontsize=16, fontweight='bold')
                ax1.set_title('ANOVA F-Statistics (Top 10 Features)', fontsize=18, fontweight='bold')
                ax1.grid(axis='x', alpha=0.25)
                ax1.tick_params(labelsize=15)
                ax1.invert_yaxis()

                colors = ['#d73027' if es == 'Large' else ('#fc8d59' if es == 'Medium' else '#fee090')
                          for es in sig_features['Effect_size']]
                ax2.barh(sig_features['Feature'], sig_features['eta_squared'],
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1.2, height=0.5)
                ax2.set_xlabel('Effect Size (η²)', fontsize=16, fontweight='bold')
                ax2.set_ylabel('Feature', fontsize=16, fontweight='bold')
                ax2.set_title('Effect Sizes by Feature', fontsize=18, fontweight='bold')
                ax2.grid(axis='x', alpha=0.25)
                ax2.tick_params(labelsize=15)
                ax2.invert_yaxis()
                plt.tight_layout()
                plt.savefig(output_dir / "anova_visualization.png", dpi=300, bbox_inches='tight')
                plt.close()

            # Feature–Structure–Condition linkage table + plot
            top_features = sig_features.head(6)
            linkage_data = []
            for _, row in top_features.iterrows():
                feat = row['Feature']
                struct_meaning = FEATURE_STRUCTURE_MAP.get(feat, 'Unknown')

                feat_correlations = []
                if getattr(trend_results, 'significant_correlations', None):
                    for corr in trend_results.significant_correlations:
                        if corr.get('feature') == feat:
                            condition = corr.get('metadata', '')
                            r_val = corr.get('correlation', 0)
                            feat_correlations.append({
                                'condition': condition,
                                'r': r_val,
                                'direction': 'increases' if r_val > 0 else 'decreases'
                            })
                feat_correlations.sort(key=lambda x: abs(x['r']), reverse=True)
                if feat_correlations:
                    top_cond = feat_correlations[0]
                    condition_effect = f"{top_cond['condition']} (r={top_cond['r']:.2f}, {top_cond['direction']})"
                else:
                    condition_effect = "No significant correlations"

                linkage_data.append({
                    'XANES_Feature': feat,
                    'Structure_Information': struct_meaning,
                    'ANOVA_Effect_Size': f"{row['eta_squared']:.3f} ({row['Effect_size']})",
                    'Primary_Condition_Effect': condition_effect,
                    'F_statistic': row['F_statistic'],
                    'p_value': row['p_value']
                })

            if linkage_data:
                linkage_df = pd.DataFrame(linkage_data)
                linkage_df.to_csv(output_dir / "feature_structure_condition_linkage.csv", index=False)

                fig, ax = plt.subplots(figsize=(18, 10))
                y_positions = np.arange(len(linkage_data))
                features = [d['XANES_Feature'] for d in linkage_data]
                structures = [d['Structure_Information'] for d in linkage_data]
                effect_sizes = [float(d['ANOVA_Effect_Size'].split()[0]) for d in linkage_data]

                colors_map = []
                for eta in effect_sizes:
                    if eta > 0.14:
                        colors_map.append('#d73027')
                    elif eta > 0.06:
                        colors_map.append('#fc8d59')
                    else:
                        colors_map.append('#fee090')

                ax.barh(y_positions, effect_sizes, color=colors_map, alpha=0.7,
                        edgecolor='black', linewidth=1.5, height=0.5)

                for i, (feat, struct) in enumerate(zip(features, structures)):
                    ax.text(-0.02, i, feat, ha='right', va='center', fontsize=16, fontweight='bold')
                    ax.text(effect_sizes[i] + 0.02, i, f'→{struct}', ha='left', va='center',
                            fontsize=16, style='italic', color='darkblue')

                ax.set_yticks([])
                ax.set_xlabel('Effect Size (η²) - Chemical Group Discrimination', fontsize=16, fontweight='bold')
                ax.set_title('XANES Features → Fe Structure Information', fontsize=18, fontweight='bold', pad=20)
                ax.set_xlim(-0.15, max(effect_sizes) * 1.5)
                ax.grid(axis='x', alpha=0.25)
                ax.tick_params(labelsize=15)
                plt.tight_layout()
                plt.savefig(output_dir / "feature_structure_linkage.png", dpi=300, bbox_inches='tight')
                plt.close()

    # 2) Condition–Structure Impact plot
    if getattr(trend_results, 'significant_correlations', None):
        corr_list = trend_results.significant_correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        categorical_impacts = []
        for corr in corr_list:
            feat = corr.get('feature', '')
            meta = corr.get('metadata', '')
            r_val = corr.get('correlation', 0)
            if meta in ['anion_type', 'ligand_type'] and feat in FEATURE_STRUCTURE_MAP:
                categorical_impacts.append({
                    'Condition': meta.replace('_', ' ').title(),
                    'Structure': FEATURE_STRUCTURE_MAP[feat],
                    'Feature': feat,
                    'Correlation': r_val
                })

        if categorical_impacts:
            cat_df = pd.DataFrame(categorical_impacts)
            for idx, condition in enumerate(['Anion Type', 'Ligand Type']):
                subset = cat_df[cat_df['Condition'] == condition]
                if len(subset) > 0:
                    y_pos = np.arange(len(subset))
                    colors = ['green' if r > 0 else 'red' for r in subset['Correlation'].values]
                    ax1.barh(y_pos + idx * (len(subset) + 1), [abs(r) for r in subset['Correlation'].values],
                             color=colors, alpha=0.6, edgecolor='black', linewidth=1.5, height=0.5)
                    for i, (_, row) in enumerate(subset.iterrows()):
                        y = i + idx * (len(subset) + 1)
                        label = f"{row['Structure']}\n({row['Feature']})"
                        ax1.text(-0.01, y, label, ha='right', va='center', fontsize=16)
                        direction_text = '→Increases' if row['Correlation'] > 0 else '→Decreases'
                        ax1.text(abs(row['Correlation']) + 0.01, y, direction_text,
                                 ha='left', va='center', fontsize=10,
                                 color='darkgreen' if row['Correlation'] > 0 else 'darkred',
                                 fontweight='bold')
                    mid_y = (len(subset) - 1) / 2 + idx * (len(subset) + 1)
                    ax1.text(0.5, mid_y, condition, ha='center', va='center',
                             fontsize=16, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            ax1.set_yticks([])
            ax1.set_xlabel('|Correlation Coefficient|', fontsize=16, fontweight='bold')
            ax1.set_title('Chemical Type Effects on Fe Structure', fontsize=17, fontweight='bold', pad=15)
            ax1.grid(axis='x', alpha=0.25)
            ax1.tick_params(labelsize=15)
            ax1.axvline(0, color='black', linewidth=1.5)
            ax1.set_xlim(-0.2, 0.7)
        else:
            ax1.text(0.5, 0.5, 'No significant categorical\ncondition correlations found',
                     ha='center', va='center', fontsize=13, transform=ax1.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax1.set_xticks([])
            ax1.set_yticks([])

        continuous_impacts = []
        for corr in corr_list:
            feat = corr.get('feature', '')
            meta = corr.get('metadata', '')
            r_val = corr.get('correlation', 0)
            if meta in ['pH', 'anion_conc', 'ligand_conc', 'conc_ratio'] and feat in FEATURE_STRUCTURE_MAP:
                continuous_impacts.append({
                    'Condition': meta.replace('_', ' ').title(),
                    'Structure': FEATURE_STRUCTURE_MAP[feat],
                    'Feature': feat,
                    'Correlation': r_val
                })

        if continuous_impacts:
            cont_df = pd.DataFrame(continuous_impacts)
            cont_df = cont_df.sort_values('Correlation', key=abs, ascending=False).head(10)
            y_pos = np.arange(len(cont_df))
            colors = ['green' if r > 0 else 'red' for r in cont_df['Correlation'].values]
            ax2.barh(y_pos, [abs(r) for r in cont_df['Correlation'].values], color=colors, alpha=0.6,
                     edgecolor='black', linewidth=1.5, height=0.5)
            for i, (_, row) in enumerate(cont_df.iterrows()):
                label = f"{row['Condition']} → {row['Structure']}"
                ax2.text(-0.01, i, label, ha='right', va='center', fontsize=16)
                direction = '→' if row['Correlation'] > 0 else '←'
                ax2.text(abs(row['Correlation']) + 0.01, i,
                         f"{direction} r={row['Correlation']:.2f}",
                         ha='left', va='center', fontsize=10,
                         color='darkgreen' if row['Correlation'] > 0 else 'darkred',
                         fontweight='bold')
            ax2.set_yticks([])
            ax2.set_xlabel('|Correlation Coefficient|', fontsize=16, fontweight='bold')
            ax2.set_title('Solution Parameter Effects on Fe Structure', fontsize=17, fontweight='bold', pad=15)
            ax2.grid(axis='x', alpha=0.25)
            ax2.tick_params(labelsize=15)
            ax2.axvline(0, color='black', linewidth=1.5)
        else:
            ax2.text(0.5, 0.5, 'No significant continuous\ncondition correlations found',
                     ha='center', va='center', fontsize=13, transform=ax2.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax2.set_xticks([])
            ax2.set_yticks([])

        plt.suptitle('Experiment Design → Fe Electronic/Coordination Structure',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / "condition_structure_impact.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3) Feature-metadata correlations (bar plot)
    if getattr(trend_results, 'significant_correlations', None):
        corr_df = pd.DataFrame(trend_results.significant_correlations)
        top_corr = corr_df.head(15) if not corr_df.empty else corr_df
        if not top_corr.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            labels = [f"{row.get('feature', 'N/A')} → {row.get('metadata', 'N/A')}"
                      for _, row in top_corr.iterrows()]
            values = top_corr.get('correlation', [0] * len(top_corr)).values
            colors = ['green' if v > 0 else 'red' for v in values]
            ax.barh(range(len(labels)), values, color=colors, alpha=0.7, height=0.55)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=16)
            ax.set_xlabel('Correlation Coefficient', fontsize=16)
            ax.set_title('Top Feature–Metadata Correlations', fontsize=18)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_dir / "feature_metadata_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()

    # 4) Condition impact analysis
    if getattr(trend_results, 'significant_correlations', None):
        corr_df = pd.DataFrame(trend_results.significant_correlations)
        if not corr_df.empty and 'metadata' in corr_df.columns:
            impact_analysis = {}
            for metadata_field in corr_df['metadata'].unique():
                field_corrs = corr_df[corr_df['metadata'] == metadata_field]
                n_significant = len(field_corrs)
                avg_correlation = field_corrs['correlation'].abs().mean()
                max_correlation = field_corrs['correlation'].abs().max()
                impact_score = n_significant * avg_correlation
                impact_analysis[metadata_field] = {
                    'count': n_significant,
                    'avg_abs_correlation': avg_correlation,
                    'max_abs_correlation': max_correlation,
                    'impact_score': impact_score
                }

            impact_df = pd.DataFrame(impact_analysis).T.sort_values('impact_score', ascending=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            colors_impact = plt.cm.RdYlGn(impact_df['impact_score'] / impact_df['impact_score'].max())
            ax1.barh(impact_df.index, impact_df['impact_score'],
                     color=colors_impact, height=0.5, edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Impact Score (Count × Avg |r|)', fontsize=16)
            ax1.set_ylabel('Experimental Condition', fontsize=16)
            ax1.set_title('Condition Impact on XANES Features', fontsize=18, fontweight='bold')
            ax1.tick_params(labelsize=15)

            x = range(len(impact_df))
            width = 0.35
            ax2.barh([i - width / 2 for i in x], impact_df['avg_abs_correlation'],
                     width, label='Avg |Correlation|', color='steelblue',
                     edgecolor='black', linewidth=1)
            ax2.barh([i + width / 2 for i in x], impact_df['max_abs_correlation'],
                     width, label='Max |Correlation|', color='coral',
                     edgecolor='black', linewidth=1)
            ax2.set_yticks(x)
            ax2.set_yticklabels(impact_df.index, fontsize=16)
            ax2.set_xlabel('Correlation Strength', fontsize=16)
            ax2.set_title('Correlation Strength by Condition', fontsize=18, fontweight='bold')
            ax2.legend(fontsize=12, loc='lower right')
            ax2.set_xlim(0, 1.0)

            plt.tight_layout()
            plt.savefig(output_dir / "condition_impact_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

            impact_df_out = impact_df.reset_index()
            impact_df_out.columns = ['Condition', 'Num_Correlations', 'Avg_Abs_Correlation', 'Max_Abs_Correlation', 'Impact_Score']
            impact_df_out = impact_df_out.sort_values('Impact_Score', ascending=False)
            impact_df_out.to_csv(output_dir / "condition_impact_ranking.csv", index=False)
