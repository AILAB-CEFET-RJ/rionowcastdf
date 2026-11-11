#!/usr/bin/env python3
"""Exporta evidências existentes das fases 0–17, sem executar diagnósticos.

Uso (na raiz do rionowcastdf, após finalizar os diagnósticos):
  python scripts/exportar_evidencias_capitulo5.py --repo .

Saída: exports/capitulo5_<UTC>/ e ZIP de mesmo nome.
O diretório contém fontes originais, CSVs derivados de Parquet (se pandas e
pyarrow estiverem disponíveis), relatório e inventário com SHA256.
Exit 0: artefatos de fases completos e hashes conferidos; exit 2: pendências.
Esse resultado NÃO certifica validade científica, nem identidade de uma cópia
do dataset; não substitui a auditoria integral. Nenhum Zarr/PNG é exportado.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

NAMES = ['structure', 'univariate', 'derived', 'joint', 'extremes',
         'seasonality', 'diurnal', 'temporal_lags', 'persistence',
         'spatial_structure', 'spatial_era5_radar', 'spatial_scales',
         'multivariate', 'regimes', 'distribution_shift', 'split_audit',
         'baselines', 'ablations']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--phases-dir', type=Path, default=Path('analysis_outputs_reference/phases_v1'))
    p.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    p.add_argument('--output', type=Path, help='Diretório NOVO; relativo à raiz do repo')
    p.add_argument('--no-csv', action='store_true', help='Não converter Parquet; originais sempre incluídos')
    args = p.parse_args()
    repo = args.repo.resolve()
    def rooted(path):
        return (repo / path).resolve() if not path.is_absolute() else path.resolve()
    phases = rooted(args.phases_dir)
    dataset = rooted(args.dataset_dir)
    if not phases.is_dir():
        p.error('Diretório de fases não encontrado: ' + str(phases))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = rooted(args.output or Path('exports') / ('capitulo5_' + stamp))
    archive = Path(str(out) + '.zip')
    if out.exists() or archive.exists():
        p.error('A saída já existe; escolha outro --output. Nada foi sobrescrito.')
    out.mkdir(parents=True)
    rows, issues, notes, phase_status = [], [], [], []

    def collect(src, rel, expected=None, phase='', required=True):
        rec = dict(phase=phase, file=str(rel), expected_sha256=expected or '',
                   exported_sha256='', size_bytes=0, status='missing')
        if not src.is_file():
            (issues if required else notes).append('Ausente: ' + str(src))
            rows.append(rec)
            return
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(dst)
        rec.update(exported_sha256=digest, size_bytes=dst.stat().st_size,
                   status='verified' if expected == digest else 'copied_unverified')
        if expected and expected != digest:
            rec['status'] = 'hash_mismatch'
            issues.append('SHA256 divergente: ' + str(src))
        rows.append(rec)
        if src.suffix == '.parquet' and not args.no_csv:
            try:
                import pandas as pd
                table = pd.read_parquet(dst)
                csv_path = out / 'tables_csv' / rel.with_suffix('.csv')
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(csv_path, index=False)
                rows.append(dict(phase=phase, file=str(csv_path.relative_to(out)),
                                 expected_sha256='', exported_sha256=sha(csv_path),
                                 size_bytes=csv_path.stat().st_size, status='derived_csv'))
            except Exception as e:
                notes.append('Conversão CSV não concluída (original preservado): ' + str(rel) + ': ' + str(e))

    for i, name in enumerate(NAMES):
        folder = f'{i:02d}_{name}'
        source = phases / folder / 'summary.json'
        collect(source, Path('phases') / folder / 'summary.json', phase=i)
        if not source.is_file():
            continue
        # Lê a cópia exportada: inventário e metadados referem-se aos mesmos bytes.
        try:
            summary = json.loads((out / 'phases' / folder / 'summary.json').read_text(encoding='utf-8'))
        except (ValueError, OSError) as e:
            issues.append(f'Fase {i}: summary inválido: {e}')
            continue
        phase_status.append({k: summary.get(k) for k in
                             ('phase', 'name', 'version', 'status', 'complete', 'passed', 'scope')})
        if summary.get('phase') != i or summary.get('version') != 'reference-phases-v1':
            issues.append(f'Fase {i}: identidade/versão inesperada; revisar antes de usar.')
        if summary.get('complete') is not True or summary.get('passed') is not True:
            issues.append(f'Fase {i}: execução não está complete/passed.')
        artifacts = summary.get('artifact_sha256', {})
        if not isinstance(artifacts, dict) or not artifacts:
            issues.append(f'Fase {i}: manifesto de hashes ausente/inválido.')
            continue
        for filename, expected in sorted(artifacts.items()):
            rel = Path(filename)
            if rel.is_absolute() or '..' in rel.parts:
                issues.append(f'Fase {i}: caminho recusado: {filename}')
                continue
            collect(phases / folder / rel, Path('phases') / folder / rel, expected, i)

    for filename in ('run_config.json', 'run_summary.json'):
        collect(phases / filename, Path('context') / filename, required=False)
    for filename in ('summary.json', 'contract.json', 'normalization.json', 'era5_identity.json'):
        collect(dataset / filename, Path('dataset_metadata') / filename, required=False)
    context = [
        'analysis_outputs_reference/integrity_full_v1/audit/summary.json',
        'analysis_outputs_reference/shared_reader_v2/summary.json',
        'configs/reference_phases.json', 'configs/phase17_channel_catalog.json',
        'configs/phase17_experiments.json',
        'docs/validation/AUTOMACAO_FASES_REFERENCIA.md',
        'src/corrdiff/diagnostics/reference_reader.py',
    ]
    context += [str(x.relative_to(repo)) for x in
                sorted((repo / 'src/corrdiff/diagnostics/reference_phases').glob('*.py'))]
    for filename in context:
        collect(repo / filename, Path('context') / filename, required=False)
    git = {}
    for label, command in [('commit', ['rev-parse', 'HEAD']),
                           ('branch', ['branch', '--show-current']),
                           ('tracked_changes', ['status', '--porcelain', '--untracked-files=no'])]:
        try:
            r = subprocess.run(['git', '-C', str(repo)] + command, capture_output=True, text=True)
            git[label] = r.stdout.strip() if r.returncode == 0 else 'unavailable'
        except OSError:
            git[label] = 'unavailable'
    write_json(out / 'git_context.json', git)
    with (out / 'inventory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['phase', 'file', 'expected_sha256', 'exported_sha256', 'size_bytes', 'status'])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(export_status='incomplete_or_inconsistent' if issues else 'phase_artifacts_verified',
                  scientific_validation=False, created_utc=stamp, phases_directory=str(phases),
                  phase_status=phase_status, issues=issues, notes=notes,
                  original_and_derived_files=len(rows),
                  note='Hashes conferem bytes com summaries locais; não autenticam uma execução por si sós. '
                       'Contexto sem hash de execução é apenas snapshot. Não usar divergências como evidência final.')
    write_json(out / 'export_report.json', report)
    (out / 'README.md').write_text(
        '# Evidências para o capítulo 5\n\n'
        'Comece por export_report.json e inventory.csv. O pacote preserva todas as saídas '
        'referenciadas pelos summaries das fases 0–17. O diretório phases contém originais; '
        'tables_csv contém conversões, quando disponíveis. NPZs preservam mapas, amostras e ajustes.\n\n'
        'Este pacote não executa análises, não acessa os PNGs/ERA5 brutos e não exporta o Zarr. '
        'Complete/passed significa conclusão das verificações implementadas, não validade meteorológica. '
        'Fase 17 contém planos, não resultados de treinamento.\n\n'
        'Para compartilhar via Git, priorize CSV/JSON e código. Não presuma que o leitor remoto '
        'consegue abrir ZIP, NPZ ou Parquet. Verifique tamanhos antes de versionar.\n', encoding='utf-8')
    shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name)
    print('Diretório:', out)
    print('ZIP:', archive)
    print('Estado:', report['export_status'])
    print('Pendências:', len(issues), '| Notas:', len(notes))
    for issue in issues:
        print('PENDÊNCIA:', issue)
    return 2 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
