#!/usr/bin/env python3
"""
CorrDiff — Consolidador das Fases 0 a 4 (compatível com Fase 2 v2 e Fase 4 v2)
=======================================

Lê SOMENTE os artefatos agregados produzidos pelas fases de análise:

  analysis_outputs/
    00_quality/
    01_univariate/
    02_derived/
    03_joint/
    04_extremes/

Não lê o train.zarr, não lê os caches do radar e não exporta amostras NPZ.
O objetivo é gerar um relatório textual compacto, adequado para inspeção
no terminal e para compartilhamento de estatísticas agregadas.

Exemplo:
    python scripts/summarize_phase0_4.py

Ou:
    python scripts/summarize_phase0_4.py \
        --analysis-root analysis_outputs \
        --output analysis_outputs/summary_phase0_4.txt \
        --top-k 8

Para somente imprimir no terminal:
    python scripts/summarize_phase0_4.py --no-save
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LINE = "=" * 88
SUBLINE = "-" * 88


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida em texto os resultados das fases 0–4 do CorrDiff."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis_outputs"),
        help="Diretório raiz dos outputs. Default: analysis_outputs",
    )
    parser.add_argument("--phase0-dir", type=Path, default=None)
    parser.add_argument("--phase1-dir", type=Path, default=None)
    parser.add_argument("--phase2-dir", type=Path, default=None)
    parser.add_argument("--phase3-dir", type=Path, default=None)
    parser.add_argument("--phase4-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo TXT de saída. Default: <analysis-root>/summary_phase0_4.txt",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Quantidade de preditores exibidos nos rankings. Default: 8",
    )
    parser.add_argument(
        "--phase4-top-k",
        type=int,
        default=5,
        help="Preditores exibidos por definição de evento na Fase 4. Default: 5",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Somente imprime no terminal; não grava TXT.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__load_error__": f"{type(exc).__name__}: {exc}"}


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"[WARN] Não foi possível ler {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def is_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "True" if value else "False"
    if is_number(value):
        x = float(value)
        if abs(x) >= 1_000_000:
            return f"{x:,.0f}"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        if abs(x) >= 100:
            return f"{x:.3f}"
        if abs(x) >= 1:
            return f"{x:.4f}"
        if x == 0:
            return "0"
        if abs(x) < 1e-4:
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 2) -> str:
    if not is_number(value):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def nested_get(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def existing_cols(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def table_text(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    index: bool = False,
) -> str:
    if df.empty:
        return "(sem dados)"
    out = df.copy()
    if columns:
        cols = existing_cols(out, columns)
        if cols:
            out = out[cols]
    if max_rows is not None:
        out = out.head(max_rows)
    return out.to_string(index=index, max_colwidth=38)


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def section(lines: list[str], title: str) -> None:
    add(lines)
    add(lines, LINE)
    add(lines, title)
    add(lines, LINE)


def subsection(lines: list[str], title: str) -> None:
    add(lines)
    add(lines, title)
    add(lines, SUBLINE)


def kv(lines: list[str], key: str, value: Any) -> None:
    add(lines, f"{key:<42}: {value}")


def summarize_phase0(lines: list[str], root: Path) -> None:
    section(lines, "FASE 0 — QUALIDADE, COBERTURA E INTEGRIDADE")

    summary = load_json(root / "dataset_summary.json")
    if not summary:
        add(lines, f"[AUSENTE] {root / 'dataset_summary.json'}")
        return

    dataset = summary.get("dataset", {})
    config = summary.get("configuration", {})
    temporal = summary.get("temporal", {})
    geometry = summary.get("patch_geometry", {})
    target_integrity = summary.get("target_integrity", {})
    mask = summary.get("mask", {})

    subsection(lines, "Estrutura do dataset")
    kv(lines, "Status da auditoria", summary.get("status", "N/A"))
    kv(lines, "Amostras / patches", fmt(dataset.get("num_samples")))
    kv(lines, "Canais de entrada", fmt(dataset.get("num_channels")))
    kv(lines, "Canais", ", ".join(map(str, dataset.get("channels", []))))
    kv(lines, "Shape input", dataset.get("input_shape"))
    kv(lines, "Shape target", dataset.get("target_shape"))
    kv(lines, "Shape mask", dataset.get("mask_shape"))
    kv(lines, "Dimensão amostral consistente", dataset.get("sample_dimension_consistent"))
    kv(lines, "Transformação do target", dataset.get("target_transform"))
    kv(lines, "Período configurado", f"{config.get('start_date')} → {config.get('end_date')}")
    kv(lines, "Frequência temporal", config.get("time_frequency"))
    kv(lines, "Grade do radar", config.get("radar_grid_shape"))
    kv(lines, "Resolução radar (km)", fmt(config.get("radar_resolution_km")))
    kv(lines, "Patch / stride", f"{config.get('patch_size')} / {config.get('stride')}")

    subsection(lines, "Cobertura temporal")
    kv(lines, "Timestamps esperados", fmt(temporal.get("expected_timestamps")))
    kv(lines, "Timestamps disponíveis", fmt(temporal.get("available_timestamps")))
    kv(lines, "Timestamps ausentes", fmt(temporal.get("missing_timestamps")))
    kv(lines, "Cobertura temporal", pct(temporal.get("coverage_ratio")))
    kv(lines, "Primeiro timestamp observado", temporal.get("first_observed_timestamp"))
    kv(lines, "Último timestamp observado", temporal.get("last_observed_timestamp"))
    kv(lines, "Patches por timestamp", temporal.get("patches_per_timestamp"))

    yearly = load_parquet(root / "yearly_coverage.parquet")
    if not yearly.empty:
        subsection(lines, "Cobertura por ano")
        add(
            lines,
            table_text(
                yearly,
                [
                    "year",
                    "expected_timestamps",
                    "available_timestamps",
                    "samples",
                    "missing_timestamps",
                    "coverage_ratio",
                    "mean_patches_per_available_timestamp",
                ],
            ),
        )

    hourly = load_parquet(root / "hourly_coverage.parquet")
    if not hourly.empty and "coverage_ratio" in hourly.columns:
        subsection(lines, "Horas UTC com menor cobertura")
        add(
            lines,
            table_text(
                hourly.sort_values("coverage_ratio").head(6),
                [
                    "hour_utc",
                    "expected_timestamps",
                    "available_timestamps",
                    "coverage_ratio",
                    "mean_patches_per_available_timestamp",
                ],
            ),
        )
        add(lines, "\nHoras UTC com maior cobertura")
        add(
            lines,
            table_text(
                hourly.sort_values("coverage_ratio", ascending=False).head(6),
                [
                    "hour_utc",
                    "expected_timestamps",
                    "available_timestamps",
                    "coverage_ratio",
                    "mean_patches_per_available_timestamp",
                ],
            ),
        )

    subsection(lines, "Geometria dos patches")
    kv(
        lines,
        "Posições esperadas por campo",
        fmt(geometry.get("expected_patch_positions_per_full_field")),
    )
    kv(
        lines,
        "Posições efetivamente observadas",
        fmt(geometry.get("actual_unique_patch_positions")),
    )
    kv(
        lines,
        "Cobertura espacial geométrica esperada",
        pct(geometry.get("expected_spatial_coverage_ratio")),
    )
    kv(
        lines,
        "Cobertura espacial observada",
        pct(geometry.get("actual_spatial_coverage_ratio")),
    )

    subsection(lines, "Integridade de máscara e target")
    kv(lines, "Pixels válidos na máscara", fmt(mask.get("valid_pixel_count")))
    kv(lines, "Razão de pixels válidos", pct(mask.get("valid_pixel_ratio")))
    kv(lines, "Violações de máscara binária", fmt(mask.get("binary_violation_count")))
    for key in (
        "negative_finite_value_count",
        "invalid_mask_region_nonzero_count",
        "nan_count_all_stored_target_pixels",
        "posinf_count_all_stored_target_pixels",
        "neginf_count_all_stored_target_pixels",
    ):
        if key in target_integrity:
            kv(lines, key, fmt(target_integrity.get(key)))

    radar_semantics = summary.get("radar_target_semantics", {})
    mapping = radar_semantics.get("target_mapping_validation", {}) if isinstance(radar_semantics, dict) else {}
    if mapping:
        subsection(lines, "Validação cache radar → target Zarr")
        kv(lines, "Status", mapping.get("status"))
        kv(lines, "Patches comparados", fmt(mapping.get("compared_samples")))
        kv(lines, "Shapes compatíveis", fmt(mapping.get("shape_compatible_samples")))
        kv(
            lines,
            "Maior erro target vs transformação esperada",
            fmt(mapping.get("max_abs_diff_stored_vs_expected_log1p")),
        )
        kv(
            lines,
            "Maior erro expm1(target) vs cache clipado",
            fmt(mapping.get("max_abs_diff_expm1_vs_clipped_cache")),
        )
        kv(lines, "Mismatches de máscara", fmt(mapping.get("total_mask_mismatch_count")))
        kv(
            lines,
            "Valores negativos na origem observados",
            fmt(mapping.get("source_negative_values_seen_in_compared_patches")),
        )

    event_rates = load_parquet(root / "target_event_rates_global.parquet")
    if not event_rates.empty:
        subsection(lines, "Desbalanceamento global do radar — distribuição de treinamento")
        add(
            lines,
            table_text(
                event_rates,
                [
                    "threshold",
                    "condition",
                    "legend_value_threshold",
                    "event_pixel_ratio",
                    "event_patch_ratio",
                    "event_pixel_count",
                    "event_patch_count",
                ],
            ),
        )

    audit_messages = load_json(root / "audit_warnings.json")
    warnings = audit_messages.get("warnings", []) if isinstance(audit_messages, dict) else []
    notes = audit_messages.get("notes", []) if isinstance(audit_messages, dict) else []
    subsection(lines, "Avisos e notas do auditor")
    if warnings:
        add(lines, "AVISOS:")
        for item in warnings:
            add(lines, f"  - {item.get('code', '?')}: {item.get('message', item.get('detail', ''))}")
    else:
        add(lines, "AVISOS: nenhum")
    if notes:
        add(lines, "NOTAS:")
        for item in notes:
            add(lines, f"  - {item.get('code', '?')}: {item.get('message', item.get('detail', ''))}")


def summarize_phase1(lines: list[str], root: Path) -> None:
    section(lines, "FASE 1 — CARACTERIZAÇÃO ESTATÍSTICA UNIVARIADA")

    summary = load_json(root / "analysis_summary.json")
    if not summary:
        add(lines, f"[AUSENTE] {root / 'analysis_summary.json'}")
        return

    dataset = summary.get("dataset", {})
    sampling = summary.get("sampling", {})
    radar_target = summary.get("radar_target", {})

    subsection(lines, "Amostragem e representatividade")
    kv(lines, "Patches do dataset", fmt(dataset.get("num_patches")))
    kv(lines, "Patches amostrados", fmt(sampling.get("sampled_patch_count")))
    kv(lines, "Estratégia", sampling.get("strategy"))
    rep = sampling.get("representativeness_vs_phase0", {})
    kv(lines, "Referência exata da Fase 0 disponível", rep.get("phase0_reference_available"))
    kv(
        lines,
        "Máx. diferença de média em unidades de σ",
        fmt(rep.get("max_abs_sample_mean_difference_in_phase0_std")),
    )
    kv(
        lines,
        "Máx. diferença relativa de std",
        pct(rep.get("max_sample_std_relative_difference")),
    )

    input_summary = load_parquet(root / "input_summary.parquet")
    if not input_summary.empty:
        subsection(lines, "Resumo dos 12 canais ERA5")
        add(
            lines,
            table_text(
                input_summary,
                [
                    "channel",
                    "expected_unit",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "sampled_skewness_from_reservoir",
                    "sampled_excess_kurtosis_from_reservoir",
                    "sample_mean_diff_in_phase0_std",
                    "sample_std_relative_diff_vs_phase0",
                ],
            ),
        )

    rh = load_parquet(root / "relative_humidity_quality.parquet")
    if not rh.empty:
        subsection(lines, "Qualidade da umidade relativa")
        add(
            lines,
            table_text(
                rh,
                [
                    "channel",
                    "below_0_ratio",
                    "above_100_ratio",
                    "outside_0_100_ratio",
                    "within_0_100_ratio",
                ],
            ),
        )

    target = load_parquet(root / "target_summary.parquet")
    if not target.empty:
        subsection(lines, "Resumo do target em seus diferentes domínios")
        add(
            lines,
            table_text(
                target,
                [
                    "channel",
                    "domain",
                    "unit",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "sampled_skewness_from_reservoir",
                    "sampled_excess_kurtosis_from_reservoir",
                ],
            ),
        )

    tq = load_parquet(root / "target_quantiles.parquet")
    if not tq.empty:
        subsection(lines, "Quantis do radar")
        preferred = {"p50", "p75", "p90", "p95", "p99", "p999"}
        qcol = "quantile" if "quantile" in tq.columns else None
        show = tq[tq[qcol].isin(preferred)] if qcol else tq
        add(
            lines,
            table_text(
                show,
                ["channel", "domain", "quantile", "probability", "value"],
            ),
        )

    threshold = load_parquet(root / "target_threshold_rates_reference.parquet")
    if threshold.empty:
        threshold = load_parquet(root / "target_threshold_rates_sampled.parquet")
    if not threshold.empty:
        subsection(lines, "Taxas por limiar do radar")
        add(
            lines,
            table_text(
                threshold,
                [
                    "threshold",
                    "legend_value_threshold",
                    "event_pixel_ratio",
                    "event_patch_ratio",
                ],
            ),
        )

    sparsity = radar_target.get("target_sparsity", {})
    if sparsity:
        subsection(lines, "Esparsidade do target")
        kv(lines, "Pixels com radar > 0", pct(sparsity.get("event_pixel_ratio_gt_0")))
        kv(lines, "Pixels secos / zero", pct(sparsity.get("dry_valid_pixel_ratio")))
        kv(lines, "Patches com ao menos um pixel > 0", pct(sparsity.get("event_patch_ratio_gt_0")))
        kv(lines, "Patches sem evento", pct(sparsity.get("dry_patch_ratio")))
        kv(lines, "Fonte", sparsity.get("source"))

    warnings = summary.get("warnings", [])
    if warnings:
        subsection(lines, "Avisos da Fase 1")
        for item in warnings:
            add(lines, f"  - {item.get('code', '?')}: {item.get('detail', '')}")


def summarize_phase2(lines: list[str], root: Path) -> None:
    section(lines, "FASE 2 — VARIÁVEIS DERIVADAS")

    summary = load_json(root / "analysis_summary.json")
    derived = load_parquet(root / "derived_summary.parquet")
    formulas = load_parquet(root / "formula_catalog.parquet")
    corr = load_parquet(root / "derived_correlations.parquet")

    if not summary and derived.empty:
        add(lines, f"[AUSENTE] resultados da Fase 2 em {root}")
        return

    if summary:
        subsection(lines, "Amostragem e validade metodológica")
        kv(lines, "Versão", summary.get("phase_version"))
        kv(lines, "Patches amostrados", fmt(summary.get("sampled_patches")))
        kv(lines, "Blocos amostrados", fmt(summary.get("sampled_blocks")))
        kv(lines, "Estratégia/fonte", summary.get("sampling_source"))
        kv(lines, "Variáveis derivadas", ", ".join(summary.get("derived_variables", [])))
        if "correlation_alignment_verified" in summary:
            kv(lines, "Correlação com pixels alinhados", summary.get("correlation_alignment_verified"))
            kv(lines, "Linhas no reservoir alinhado", fmt(summary.get("aligned_reservoir_rows")))
            kv(lines, "Nota de correlação", summary.get("correlation_note"))
        kv(lines, "Observação metodológica", summary.get("note"))

    if not formulas.empty:
        subsection(lines, "Catálogo das variáveis derivadas")
        add(lines, table_text(formulas, ["name", "formula", "unit", "interpretation"]))

    if not derived.empty:
        subsection(lines, "Estatísticas das variáveis derivadas")
        add(
            lines,
            table_text(
                derived,
                [
                    "name",
                    "unit",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "skewness_reservoir",
                    "excess_kurtosis_reservoir",
                    "aligned_reservoir_count",
                ],
            ),
        )

    if not corr.empty and {"var_a", "var_b", "pearson_r"}.issubset(corr.columns):
        unique = corr[corr["var_a"] < corr["var_b"]].copy()
        unique["abs_r"] = unique["pearson_r"].abs()
        subsection(lines, "Maiores correlações entre variáveis derivadas — Pearson")
        add(
            lines,
            table_text(
                unique.sort_values("abs_r", ascending=False).head(12),
                ["var_a", "var_b", "pearson_r", "spearman_rho", "n_aligned"],
            ),
        )
        if "spearman_rho" in unique.columns:
            unique["abs_rho"] = unique["spearman_rho"].abs()
            subsection(lines, "Maiores correlações entre variáveis derivadas — Spearman")
            add(
                lines,
                table_text(
                    unique.sort_values("abs_rho", ascending=False).head(12),
                    ["var_a", "var_b", "spearman_rho", "pearson_r", "n_aligned"],
                ),
            )


def summarize_phase3(lines: list[str], root: Path, top_k: int) -> None:
    section(lines, "FASE 3 — RELAÇÕES ERA5 × RADAR")

    summary = load_json(root / "analysis_summary.json")
    metrics = load_parquet(root / "association_metrics.parquet")
    bins = load_parquet(root / "conditional_bins.parquet")

    if not summary and metrics.empty:
        add(lines, f"[AUSENTE] resultados da Fase 3 em {root}")
        return

    if summary:
        subsection(lines, "Amostragem conjunta")
        kv(lines, "Observações pixel-a-pixel retidas", fmt(summary.get("retained_pixel_observations")))
        kv(lines, "Preditores", fmt(summary.get("predictor_count")))
        kv(lines, "Preditores ERA5 brutos", fmt(summary.get("raw_predictor_count")))
        kv(lines, "Preditores derivados", fmt(summary.get("derived_predictor_count")))
        kv(lines, "Taxa de radar > 0 na amostra", pct(summary.get("positive_event_rate_in_sample")))
        kv(lines, "Domínio radar", summary.get("radar_domain"))
        kv(lines, "Nota", summary.get("note"))

    if not metrics.empty:
        subsection(lines, f"Top {top_k} — |Pearson| com intensidade do radar (todos os pixels válidos)")
        t = ranked_metric(metrics, "pearson_r", top_k)
        add(lines, table_text(t, ["predictor", "source", "pearson_r", "spearman_rho", "mutual_information_regression"]))

        subsection(lines, f"Top {top_k} — |Spearman| com intensidade do radar")
        t = ranked_metric(metrics, "spearman_rho", top_k)
        add(lines, table_text(t, ["predictor", "source", "spearman_rho", "pearson_r", "mutual_information_regression"]))

        subsection(lines, f"Top {top_k} — Mutual information com intensidade do radar")
        t = ranked_metric(metrics, "mutual_information_regression", top_k, absolute=False)
        add(lines, table_text(t, ["predictor", "source", "mutual_information_regression", "pearson_r", "spearman_rho"]))

        subsection(lines, f"Top {top_k} — associação com ocorrência radar > 0")
        t = ranked_metric(metrics, "point_biserial_r_positive", top_k)
        add(
            lines,
            table_text(
                t,
                [
                    "predictor",
                    "source",
                    "point_biserial_r_positive",
                    "mutual_information_positive_event",
                    "positive_event_rate",
                ],
            ),
        )

        subsection(lines, f"Top {top_k} — relações somente entre pixels positivos")
        t = ranked_metric(metrics, "spearman_rho_positive", top_k)
        add(
            lines,
            table_text(
                t,
                [
                    "predictor",
                    "source",
                    "pearson_r_positive",
                    "spearman_rho_positive",
                    "mutual_information_regression_positive",
                    "n_positive",
                ],
            ),
        )

        subsection(lines, "Tabela completa de métricas de associação")
        add(
            lines,
            table_text(
                metrics.sort_values(
                    "mutual_information_regression",
                    ascending=False,
                    na_position="last",
                ),
                [
                    "predictor",
                    "source",
                    "pearson_r",
                    "spearman_rho",
                    "mutual_information_regression",
                    "point_biserial_r_positive",
                    "mutual_information_positive_event",
                    "pearson_r_positive",
                    "spearman_rho_positive",
                    "mutual_information_regression_positive",
                ],
            ),
        )

    if not bins.empty:
        subsection(lines, "Curvas condicionais — amplitude do efeito entre primeiro e último bin")
        rows: list[dict[str, Any]] = []
        for predictor, group in bins.groupby("predictor", sort=False):
            g = group.sort_values("bin_index")
            if len(g) < 2:
                continue
            first = g.iloc[0]
            last = g.iloc[-1]
            rows.append(
                {
                    "predictor": predictor,
                    "radar_mean_delta_last_minus_first": last.get("radar_mean", math.nan) - first.get("radar_mean", math.nan),
                    "positive_rate_delta": last.get("positive_rate", math.nan) - first.get("positive_rate", math.nan),
                    "ge20_rate_delta": last.get("ge20_rate", math.nan) - first.get("ge20_rate", math.nan),
                    "ge30_rate_delta": last.get("ge30_rate", math.nan) - first.get("ge30_rate", math.nan),
                    "ge40_rate_delta": last.get("ge40_rate", math.nan) - first.get("ge40_rate", math.nan),
                }
            )
        if rows:
            delta = pd.DataFrame(rows)
            delta["__score"] = delta["ge30_rate_delta"].abs()
            add(
                lines,
                table_text(
                    delta.sort_values("__score", ascending=False).drop(columns="__score").head(top_k),
                    [
                        "predictor",
                        "radar_mean_delta_last_minus_first",
                        "positive_rate_delta",
                        "ge20_rate_delta",
                        "ge30_rate_delta",
                        "ge40_rate_delta",
                    ],
                ),
            )


def summarize_phase4(lines: list[str], root: Path, top_k: int) -> None:
    section(lines, "FASE 4 — EVENTOS INTENSOS E EXTREMOS")

    summary = load_json(root / "analysis_summary.json")
    prevalence = load_parquet(root / "event_prevalence.parquet")
    effects = load_parquet(root / "effect_sizes.parquet")
    cond = load_parquet(root / "conditional_predictor_stats.parquet")
    deciles = load_parquet(root / "event_rate_by_predictor_decile.parquet")
    strata = load_parquet(root / "stratum_sampling.parquet")

    if not summary and prevalence.empty:
        add(lines, f"[AUSENTE] resultados da Fase 4 em {root}")
        return

    if summary:
        subsection(lines, "Configuração, amostragem e pesos")
        kv(lines, "Versão", summary.get("phase_version"))
        kv(lines, "Fonte da amostra", summary.get("sample_source"))
        kv(lines, "Blocos amostrados", fmt(summary.get("sampled_blocks")))
        kv(lines, "Patches amostrados", fmt(summary.get("sampled_patches")))
        kv(
            lines,
            "Pixels válidos vistos nos blocos",
            fmt(summary.get("valid_population_pixels_seen_in_sampled_blocks")),
        )
        kv(
            lines,
            "Linhas retidas na amostra estratificada",
            fmt(summary.get("retained_stratified_sample_rows", summary.get("observations"))),
        )
        kv(lines, "Preditores", fmt(summary.get("predictor_count")))
        kv(
            lines,
            "Taxa radar > 0 ponderada",
            pct(summary.get("positive_radar_rate_weighted_sampled_blocks", summary.get("positive_radar_rate"))),
        )
        if "positive_radar_rate_unweighted_retained_sample" in summary:
            kv(
                lines,
                "Taxa radar > 0 NÃO ponderada na amostra",
                pct(summary.get("positive_radar_rate_unweighted_retained_sample")),
            )
        kv(lines, "Definições de evento", fmt(summary.get("threshold_count")))
        kv(lines, "Eventos degenerados", summary.get("degenerate_event_definitions"))
        kv(lines, "Referência exata Fase 0 disponível", summary.get("phase0_reference_available"))
        if "max_abs_fixed_threshold_rate_diff_vs_phase0" in summary:
            kv(
                lines,
                "Máx. |taxa ponderada - taxa exata Fase 0|",
                fmt(summary.get("max_abs_fixed_threshold_rate_diff_vs_phase0")),
            )
        kv(lines, "Método de ponderação", summary.get("weighting_method"))
        kv(lines, "Domínio radar", summary.get("radar_domain"))
        kv(lines, "Nota", summary.get("interpretation_note"))

    if not strata.empty:
        subsection(lines, "Amostragem por estrato de intensidade")
        add(
            lines,
            table_text(
                strata,
                [
                    "stratum",
                    "lower",
                    "upper",
                    "population_count_in_sampled_blocks",
                    "sample_count",
                    "sampling_fraction",
                    "inverse_sampling_weight",
                ],
            ),
        )

    if not prevalence.empty:
        subsection(lines, "Prevalência das definições de evento")
        add(
            lines,
            table_text(
                prevalence,
                [
                    "event_id",
                    "family",
                    "threshold",
                    "operator",
                    "sample_event_count_unweighted",
                    "effective_sample_size_event",
                    "weighted_event_rate_sampled_blocks",
                    "reference_event_rate_phase0",
                    "event_rate",
                    "event_rate_source",
                    "event_count",
                    "event_rate",
                    "non_event_count",
                ],
            ),
        )

    if not effects.empty and "standardized_mean_difference" in effects.columns:
        subsection(lines, "Preditores que mais diferenciam evento × não-evento")
        event_order = (
            prevalence["event_id"].tolist()
            if "event_id" in prevalence.columns
            else effects["event_id"].drop_duplicates().tolist()
        )
        for event_id in event_order:
            e = effects[effects["event_id"] == event_id].copy()
            if e.empty:
                continue
            e["abs_smd"] = e["standardized_mean_difference"].abs()
            e = e.sort_values("abs_smd", ascending=False).head(top_k)
            add(lines)
            add(lines, f"[{event_id}]  top {top_k} por |standardized_mean_difference|")
            add(
                lines,
                table_text(
                    e,
                    [
                        "predictor",
                        "source",
                        "event_mean",
                        "non_event_mean",
                        "mean_difference",
                        "standardized_mean_difference",
                        "n_event_sample",
                        "effective_n_event",
                        "n_event",
                    ],
                ),
            )

        subsection(lines, "Resumo máximo por preditor ao longo das definições de evento")
        e = effects.copy()
        e["abs_smd"] = e["standardized_mean_difference"].abs()
        idx = e.groupby("predictor")["abs_smd"].idxmax()
        strongest = e.loc[idx].sort_values("abs_smd", ascending=False)
        add(
            lines,
            table_text(
                strongest,
                [
                    "predictor",
                    "source",
                    "event_id",
                    "threshold",
                    "event_mean",
                    "non_event_mean",
                    "standardized_mean_difference",
                    "effective_n_event",
                ],
            ),
        )

    if not deciles.empty:
        subsection(lines, "Monotonicidade aproximada por decil do preditor")
        rows = []
        for (event_id, predictor), g in deciles.groupby(["event_id", "predictor"], sort=False):
            g = g.sort_values("decile")
            if len(g) < 2:
                continue
            first = g.iloc[0]
            last = g.iloc[-1]
            rows.append(
                {
                    "event_id": event_id,
                    "predictor": predictor,
                    "rate_decile_1": first.get("event_rate", math.nan),
                    "rate_decile_last": last.get("event_rate", math.nan),
                    "delta_last_minus_first": last.get("event_rate", math.nan) - first.get("event_rate", math.nan),
                    "weighted": last.get("weighted", None),
                }
            )
        if rows:
            dd = pd.DataFrame(rows)
            dd["abs_delta"] = dd["delta_last_minus_first"].abs()
            add(
                lines,
                table_text(
                    dd.sort_values("abs_delta", ascending=False).head(max(20, top_k * 2)),
                    [
                        "event_id",
                        "predictor",
                        "rate_decile_1",
                        "rate_decile_last",
                        "delta_last_minus_first",
                        "weighted",
                    ],
                ),
            )

    if not cond.empty:
        focus = cond[
            cond["event_id"].isin(["fixed_ge_30", "fixed_ge_40", "fixed_ge_45"])
        ]
        if not focus.empty:
            subsection(lines, "Medianas condicionais — eventos fixos mais intensos")
            pivot = focus.pivot_table(
                index=["event_id", "predictor"],
                columns="group",
                values="median",
                aggfunc="first",
            ).reset_index()
            if {"event", "non_event"}.issubset(pivot.columns):
                pivot["median_difference"] = pivot["event"] - pivot["non_event"]
                pivot["abs_diff"] = pivot["median_difference"].abs()
                add(
                    lines,
                    table_text(
                        pivot.sort_values("abs_diff", ascending=False).head(max(20, top_k * 2)),
                        ["event_id", "predictor", "event", "non_event", "median_difference"],
                    ),
                )


def summarize_integrated(lines: list[str], dirs: dict[int, Path], top_k: int) -> None:
    section(lines, "SÍNTESE CRUZADA AUTOMÁTICA — FASES 0 A 4")
    add(
        lines,
        "Esta seção não substitui interpretação científica; ela apenas cruza métricas "
        "agregadas produzidas pelos scripts anteriores.",
    )

    p3 = load_parquet(dirs[3] / "association_metrics.parquet")
    p4 = load_parquet(dirs[4] / "effect_sizes.parquet")

    if not p3.empty:
        subsection(lines, "Preditores com evidência em múltiplas métricas da Fase 3")
        score_df = p3.copy()
        metric_cols = [
            "pearson_r",
            "spearman_rho",
            "mutual_information_regression",
            "point_biserial_r_positive",
            "mutual_information_positive_event",
            "spearman_rho_positive",
            "mutual_information_regression_positive",
        ]
        ranks = []
        for metric in metric_cols:
            if metric not in score_df.columns:
                continue
            vals = pd.to_numeric(score_df[metric], errors="coerce")
            basis = vals.abs() if "mutual_information" not in metric else vals
            rank = basis.rank(method="average", ascending=False, na_option="bottom")
            ranks.append(rank)
        if ranks:
            score_df["mean_rank_phase3"] = pd.concat(ranks, axis=1).mean(axis=1)
            add(
                lines,
                table_text(
                    score_df.sort_values("mean_rank_phase3").head(top_k),
                    [
                        "predictor",
                        "source",
                        "mean_rank_phase3",
                        "pearson_r",
                        "spearman_rho",
                        "mutual_information_regression",
                        "point_biserial_r_positive",
                        "spearman_rho_positive",
                    ],
                ),
            )

    if not p4.empty and "standardized_mean_difference" in p4.columns:
        subsection(lines, "Preditores mais recorrentes entre os extremos da Fase 4")
        tmp = p4.copy()
        tmp["abs_smd"] = tmp["standardized_mean_difference"].abs()
        agg = (
            tmp.groupby(["predictor", "source"], as_index=False)
            .agg(
                max_abs_smd=("abs_smd", "max"),
                mean_abs_smd=("abs_smd", "mean"),
                median_abs_smd=("abs_smd", "median"),
                event_definitions=("event_id", "nunique"),
            )
            .sort_values(["mean_abs_smd", "max_abs_smd"], ascending=False)
        )
        add(
            lines,
            table_text(
                agg.head(top_k),
                [
                    "predictor",
                    "source",
                    "mean_abs_smd",
                    "median_abs_smd",
                    "max_abs_smd",
                    "event_definitions",
                ],
            ),
        )

    if not p3.empty and not p4.empty:
        subsection(lines, "Cruzamento Fase 3 × Fase 4")
        p3x = p3.copy()
        p4x = p4.copy()
        p4x["abs_smd"] = p4x["standardized_mean_difference"].abs()
        eagg = (
            p4x.groupby("predictor", as_index=False)
            .agg(mean_abs_smd=("abs_smd", "mean"), max_abs_smd=("abs_smd", "max"))
        )
        merged = p3x.merge(eagg, on="predictor", how="inner")
        if not merged.empty:
            # Purely descriptive composite: percentile ranks averaged.
            components = []
            for col in ["spearman_rho", "mutual_information_regression", "point_biserial_r_positive", "mean_abs_smd"]:
                if col not in merged.columns:
                    continue
                v = pd.to_numeric(merged[col], errors="coerce")
                if col in {"spearman_rho", "point_biserial_r_positive"}:
                    v = v.abs()
                components.append(v.rank(pct=True))
            if components:
                merged["descriptive_cross_phase_score"] = pd.concat(components, axis=1).mean(axis=1)
                merged = merged.sort_values("descriptive_cross_phase_score", ascending=False)
                add(
                    lines,
                    "ATENÇÃO: 'descriptive_cross_phase_score' é apenas um índice exploratório "
                    "de percentis, não uma medida de importância causal nem critério definitivo de seleção.",
                )
                add(
                    lines,
                    table_text(
                        merged.head(top_k),
                        [
                            "predictor",
                            "source",
                            "spearman_rho",
                            "mutual_information_regression",
                            "point_biserial_r_positive",
                            "mean_abs_smd",
                            "max_abs_smd",
                            "descriptive_cross_phase_score",
                        ],
                    ),
                )


def main() -> int:
    args = parse_args()
    root = args.analysis_root.expanduser()

    dirs = {
        0: args.phase0_dir or root / "00_quality",
        1: args.phase1_dir or root / "01_univariate",
        2: args.phase2_dir or root / "02_derived",
        3: args.phase3_dir or root / "03_joint",
        4: args.phase4_dir or root / "04_extremes",
    }
    dirs = {k: Path(v).expanduser() for k, v in dirs.items()}

    lines: list[str] = []
    add(lines, LINE)
    add(lines, "CORRDIFF — CONSOLIDAÇÃO DOS RESULTADOS DAS FASES 0 A 4")
    add(lines, LINE)
    add(lines, "Relatório composto somente por estatísticas agregadas dos analysis_outputs.")
    add(lines, "O script não lê o Zarr, caches do radar ou arquivos NPZ com amostras.")
    add(lines)
    for phase, directory in dirs.items():
        add(lines, f"Fase {phase}: {directory}")

    summarize_phase0(lines, dirs[0])
    summarize_phase1(lines, dirs[1])
    summarize_phase2(lines, dirs[2])
    summarize_phase3(lines, dirs[3], max(1, args.top_k))
    summarize_phase4(lines, dirs[4], max(1, args.phase4_top_k))
    summarize_integrated(lines, dirs, max(1, args.top_k))

    section(lines, "FIM DA CONSOLIDAÇÃO")
    add(lines, "Para revisão externa, copie o conteúdo deste relatório textual.")
    add(lines, "Evite compartilhar train.zarr, caches, NPZ de amostras ou dados brutos se a política corporativa não permitir.")

    text = "\n".join(lines) + "\n"
    print(text)

    if not args.no_save:
        output = args.output or (root / "summary_phase0_4.txt")
        output = output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"\n[OK] Relatório salvo em: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
