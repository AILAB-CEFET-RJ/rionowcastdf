# Validação de integridade do Zarr de referência

Copie `audit_reference_zarr.py` para a raiz do repositório. Não substitui o builder nem modifica seu contrato. Execute no mesmo ambiente usado para construir o Zarr, com os diretórios originais ainda disponíveis.

## Execução no CEFET

```bash
cd /home/vsantos/repositories/rionowcastdf
source .venv-zarr-reference/bin/activate
export RADAR_ROOT=/home/vsantos/data/radar_sumare
export ERA5_ROOT=/home/vsantos/data/reanalysis/cds/era5
mkdir -p logs
nohup python -u audit_reference_zarr.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --output-dir analysis_outputs_restored/reference_integrity_v1 \
  > logs/reference_integrity_v1.log 2>&1 &
echo "PID: $!"
```

```bash
tail -n 60 -f logs/reference_integrity_v1.log
```

O diretório de saída precisa ser novo. Em nova execução use, por exemplo, `reference_integrity_v2`. Não execute simultaneamente com uma construção ou alteração dos arquivos de entrada. Não remova a pasta `staging` do dataset antes da auditoria.

## Escopo

- Verifica hashes dos artefatos, contrato/configuração/código, manifesto congelado e manifesto do dataset.
- Verifica horários UTC, nomes locais com horário de verão, tolerância temporal, partições e purge conforme contrato do builder. Caminhos de aquisição não podem cruzar partições; imagens transparentes idênticas em datas distintas não são automaticamente tratadas como vazamento.
- Verifica hashes das grades horárias em `staging` e recalcula a normalização com todas as horas de treino, antes dos patches; compara JSON e NPZ.
- Lê todos os patches dos quatro Zarrs: shapes, dtypes, dimensões, ausência de fill-value que mascare zeros válidos, valores finitos, máscara, posições, horário fonte, deslocamento e SHA da imagem.
- Compara cada patch de entrada com a grade ERA5 interpolada preservada em staging; compara cada alvo e máscara com o cache do radar. Não arredonda valores do alvo para aceitar classes próximas.
- Recalcula todos os hashes lógicos dos arrays, contagens de patches, suporte e pixels sem eco.
- Decodifica até duas imagens de conteúdo distinto por combinação split/política, distribuídas temporalmente, e compara com o cache. Lista a amostra em `png_samples.json`. Pode ajustar `--png-samples-per-policy`.

## Resultados

`summary.json` registra o progresso e, no sucesso, `complete: true` e `integrity_passed: true`. Erros encerram com código diferente de zero e ficam registrados no resumo e log. O programa interrompe na primeira inconsistência, sem reparar ou alterar os dados.

A verificação integral lê todos os tensores e pode demorar devido ao volume descomprimido. Progresso a cada 500 horas por split; a checagem inicial de staging/normalização também pode levar algum tempo.

PASS significa consistência técnica com o contrato e os artefatos preservados. Não confirma cobertura física do radar, não executa CorrDiff, não refaz independentemente a interpolação dos Parquets ERA5 e não substitui os diagnósticos científicos. A comparação PNG/cache é amostral e usa o decodificador contratado. A seleção nearest2 é confrontada com o manifesto congelado; não se refaz a busca de candidatos no acervo inteiro.

## Testes locais

O teste usa o fixture do builder em `tests/test_reference_zarr.py`:

```bash
python -m unittest discover -s tests -p test_reference_integrity.py -v
```

Foram verificados um dataset sintético completo, a detecção de alteração numérica em um patch e a recusa de gravar relatórios dentro do dataset.
