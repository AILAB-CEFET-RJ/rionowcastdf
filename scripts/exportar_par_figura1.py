#!/usr/bin/env python3
"""Exporta um par contemporâneo do reference-zarr-v1, sem alterar o dataset.
Dependências: numpy e zarr já utilizados no ambiente dataset.
"""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reconstruct(patches, rows, cols, shape, name):
    """Copia valores originais e exige igualdade exata em sobreposições."""
    require(len(patches) == len(rows) == len(cols), f'{name}: índices incompatíveis')
    out = np.zeros(shape, dtype=patches.dtype)
    count = np.zeros(shape, dtype=np.uint16)
    ph, pw = patches.shape[-2:]
    for patch, row, col in zip(patches, rows, cols):
        r, c = int(row), int(col)
        require(0 <= r <= shape[0]-ph and 0 <= c <= shape[1]-pw,
                f'{name}: patch fora dos limites: {(r, c)}')
        dest = out[r:r+ph, c:c+pw]
        n = count[r:r+ph, c:c+pw]
        overlap = n > 0
        require(np.array_equal(dest[overlap], patch[overlap]),
                f'{name}: valores divergentes na sobreposição em {(r, c)}')
        dest[~overlap] = patch[~overlap]
        n += 1
    require(np.all(count > 0), f'{name}: patches não cobrem toda a grade; exportação interrompida')
    return out, count


def read_radar(g, hour, per_hour, shape, support):
    a, b = hour*per_hour, (hour+1)*per_hour
    ts = np.asarray(g['timestamps'][a:b])
    require(len(ts) == per_hour and np.all(ts == ts[0]),
            'Bloco não corresponde a um único horário; não assumir 48 patches neste dataset')
    if a:
        require(int(g['timestamps'][a-1]) != int(ts[0]), 'Horário começa antes do bloco')
    if b < g['timestamps'].shape[0]:
        require(int(g['timestamps'][b]) != int(ts[0]), 'Horário continua depois do bloco')
    rows = np.asarray(g['patch_row'][a:b])
    cols = np.asarray(g['patch_col'][a:b])
    require(len(set(zip(rows.tolist(), cols.tolist()))) == per_hour, 'Posições repetidas no horário')
    target, count = reconstruct(np.asarray(g['target'][a:b, 0]), rows, cols, shape, 'target')
    mask, _ = reconstruct(np.asarray(g['mask'][a:b, 0]), rows, cols, shape, 'mask')
    require(mask.dtype == np.bool_, 'Máscara deve ser booleana')
    require(not np.any(mask & ~support), 'Máscara válida fora do suporte')
    require(np.any(mask), 'Horário sem pixels válidos')
    require(np.isfinite(target).all(), 'Alvo contém NaN/Inf')
    require(np.all(target[~mask] == 0), 'Placeholder desconhecido diferente de zero')
    scaled = target.astype(np.float64)*50.0
    rounded = np.rint(scaled)
    require(np.all(np.abs(scaled[mask]-rounded[mask]) < 1e-4), 'Alvo incompatível com classes / 50')
    require(np.isin(rounded[mask], [0,20,25,30,35,40,45,50]).all(), 'Classes inesperadas')
    code = rounded.astype(np.uint8)
    code[~mask] = 0
    return dict(a=a, b=b, timestamp=int(ts[0]), rows=rows, cols=cols,
                target=target, mask=mask, code=code, count=count,
                echo_fraction=float(np.mean(code[mask] > 0)),
                ge35_pixels=int(np.count_nonzero(mask & (code >= 35))),
                max_code=int(code[mask].max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    ap.add_argument('--output', type=Path, default=Path('exports/figura1'))
    ap.add_argument('--candidates', type=int, default=128,
                    help='Máximo de horários distribuídos uniformemente ao longo do treino')
    ap.add_argument('--hour-index', type=int, help='Índice zero-based de um horário específico do treino')
    args = ap.parse_args()
    require(args.candidates > 0, '--candidates deve ser positivo')
    import zarr
    root = args.dataset.resolve()
    summary = json.loads((root/'summary.json').read_text())
    require(summary.get('complete') is True, 'Dataset não marcado como completo')
    require(summary.get('target_transform') == 'target_code / 50', 'Transformação inesperada')
    g = zarr.open_group(str(root/'train.zarr'), mode='r')
    attrs = dict(g.attrs)
    require(attrs.get('complete') is True, 'Treino não marcado como completo')
    require(attrs.get('input_normalized') is False, 'Entrada não confirmada em unidades físicas')
    require(attrs.get('target_transform') == 'target_code / 50', 'Transformação do grupo inesperada')
    require(attrs.get('timestamp_units') == 'seconds since 1970-01-01T00:00:00Z', 'Unidade temporal inesperada')
    channels = list(attrs['channels'])
    channel = channels.index('tcwv')
    n = g['timestamps'].shape[0]
    hours = int(summary['splits']['train']['hours_with_patches'])
    require(hours > 0 and n == int(summary['splits']['train']['patches']), 'Contagem inconsistente')
    # Este exportador é específico do contrato fornecido: 48 posições por horário.
    per_hour = 48
    require(n == hours*per_hour, 'O dataset não possui a estrutura esperada de 48 patches por horário')
    lat = np.asarray(g['grid_lat'][:])
    lon = np.asarray(g['grid_lon'][:])
    support = np.asarray(g['grid_support'][:], dtype=bool)
    require(lat.shape == lon.shape == support.shape == (70,84), 'Grade inesperada')
    require(np.isfinite(lat).all() and np.isfinite(lon).all(), 'Coordenadas não finitas')
    require(g['input'].shape == (n,len(channels),32,32), 'Shape de input inesperado')
    require(g['target'].shape == g['mask'].shape == (n,1,32,32), 'Shapes de target/mask inesperados')
    if args.hour_index is not None:
        require(0 <= args.hour_index < hours, '--hour-index fora dos limites')
        candidates = [args.hour_index]
    else:
        candidates = np.unique(np.linspace(0,hours-1,min(args.candidates,hours),dtype=int)).tolist()
    print(f'Amostrando até {len(candidates)} horários do treinamento; fonte em somente leitura.', flush=True)
    selected = None
    scanned = []
    threshold_met = False
    for k,hour in enumerate(candidates,1):
        item = read_radar(g,hour,per_hour,lat.shape,support)
        scanned.append({'hour_index':int(hour),'timestamp':item['timestamp'],
                        'echo_fraction':item['echo_fraction'],'ge35_pixels':item['ge35_pixels'],
                        'max_code':item['max_code']})
        item['hour_index'] = int(hour)
        if selected is None or item['echo_fraction'] > selected['echo_fraction']:
            selected = item
        if args.hour_index is not None or (item['echo_fraction'] >= .05 and item['ge35_pixels'] >= 3):
            selected = item
            threshold_met = item['echo_fraction'] >= .05 and item['ge35_pixels'] >= 3
            break
        if k % 16 == 0:
            print(f'{k}/{len(candidates)} horários verificados; maior fração de eco: {selected["echo_fraction"]:.2%}',flush=True)
    require(selected is not None and selected['echo_fraction'] > 0,
            'Nenhum eco encontrado na amostra. Execute com --candidates 512.')
    a,b = selected['a'],selected['b']
    tcwv,count = reconstruct(np.asarray(g['input'][a:b,channel,:,:]),
                            selected['rows'],selected['cols'],lat.shape,'tcwv')
    require(np.isfinite(tcwv).all() and np.all(tcwv >= 0), 'tcwv não finito ou negativo')
    require(np.array_equal(count,selected['count']), 'Cobertura de patches inconsistente')
    def constant(name):
        v = np.asarray(g[name][a:b])
        require(np.all(v == v[0]), f'{name} varia dentro do horário')
        return v[0]
    src = int(constant('source_timestamps'))
    offset = int(constant('offset_seconds'))
    require(src-selected['timestamp'] == offset and abs(offset) <= 120,'Deslocamento temporal inconsistente')
    sha = constant('source_sha256')
    sha = bytes(sha).decode('ascii') if isinstance(sha,(bytes,np.bytes_)) else str(sha)
    require(len(sha) == 64 and all(c in '0123456789abcdef' for c in sha.lower()),'Hash de origem inválido')
    utc = lambda value: datetime.fromtimestamp(value,timezone.utc).isoformat()
    metadata = {
        'export_version':'figura1-pair-v1','split':'train',
        'hour_index':selected['hour_index'],'patch_start':a,'patch_stop_exclusive':b,
        'patches':b-a,'grid_shape':list(lat.shape),
        'era5_timestamp_utc':utc(selected['timestamp']),'radar_timestamp_utc':utc(src),
        'offset_seconds':offset,'source_sha256':sha,
        'era5_channel':'tcwv','era5_unit':'kg m-2',
        'era5_representation':'Campo já interpolado na grade do dataset; não é a grade nativa ERA5',
        'radar_representation':'Classes exibidas; código 0 = ausência de eco exibido, não 0 dBZ medido',
        'unknown_rule':'mask=false; ignore target_code=0 nestas posições',
        'coverage_state':attrs.get('coverage_state'),
        'spatial_validity_confirmed':attrs.get('spatial_validity_confirmed'),
        'selection_rule':('explicit hour index' if args.hour_index is not None else
            'Primeiro candidato com >=5% de eco e >=3 pixels na classe >=35; fallback: maior fração de eco entre candidatos'),
        'selection_purpose':'Ilustração; não é seleção aleatória nem demonstra representatividade ou desempenho',
        'selection_threshold_met':threshold_met,'candidate_limit':len(candidates),
        'candidates_examined':scanned,'echo_fraction_valid_unique_pixels':selected['echo_fraction'],
        'valid_unique_pixels':int(selected['mask'].sum()),'maximum_display_code':selected['max_code'],
        'overlap_policy':'Igualdade exata verificada; cópia sem médias ou interpolação adicional',
        'group_attributes':attrs,
        'artifact_sha256_from_summary':summary.get('artifact_sha256',{}),
        'note':'Metadados preservados; hashes em campos distintos não são presumidos equivalentes.'
    }
    buf = io.BytesIO()
    np.savez_compressed(buf,tcwv=tcwv,target_normalized=selected['target'],
        target_code=selected['code'],mask=selected['mask'],grid_lat=lat,grid_lon=lon,
        grid_support=support,patch_overlap_count=count,
        era5_timestamp=np.int64(selected['timestamp']),radar_timestamp=np.int64(src),
        offset_seconds=np.int16(offset),source_sha256=np.asarray(sha))
    payload=buf.getvalue()
    metadata['pair_npz_sha256']=hashlib.sha256(payload).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.fromtimestamp(selected['timestamp'],timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=args.output/f'par_figura1_{stamp}.zip'
    # Modo x impede sobrescrita silenciosa de uma exportação anterior.
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('par_figura1.npz',payload)
        z.writestr('metadados.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        for name in ['contract.json','summary.json','normalization.json']:
            file=root/name
            if file.is_file():z.write(file,'proveniencia/'+name)
        z.writestr('LEIA-ME.txt',
            'Par real exportado do treino para ilustração da Figura 1.\n'
            'TCWV em kg m-2 na grade interpolada. Sem reconstrução da grade nativa.\n'
            'Radar quantizado: use target_code junto com mask; zero válido indica ausência de eco exibido.\n'
            'A grade geométrica não comprova cobertura observacional efetiva.\n'
            'Nenhuma média foi aplicada aos patches sobrepostos.\n'
            'Consulte metadados.json para horários, seleção e proveniência.\n')
    print(f'ERA5: {utc(selected["timestamp"])}',flush=True)
    print(f'Radar: {utc(src)} | deslocamento: {offset:+d} s',flush=True)
    print(f'Eco nos pixels válidos: {selected["echo_fraction"]:.2%}',flush=True)
    print(f'ARQUIVO PARA ENVIAR: {output.resolve()} ({output.stat().st_size/1024:.1f} KiB)',flush=True)

if __name__ == '__main__':
    main()
