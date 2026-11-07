# Auditoria integral do dataset de referência

## O que esta entrega executa

A versão 2 verifica todos os tensores dos quatro Zarrs, todos os objetos de cache usados, todos os caminhos PNG selecionados no manifesto e todas as horas/canais ERA5 selecionados, relidos dos Parquets. O lançador ativa os modos completos automaticamente.

Não é uma nova auditoria de todos os PNGs do acervo 2011–2024: essa é a finalidade da 14m. O universo aqui é o dataset selecionado. PNGs distintos com bytes idênticos têm cada caminho verificado por SHA-256; seu conteúdo é decodificado uma vez por hash/política. Isso evita repetir decodificações sem deixar caminhos selecionados sem verificação.

O recálculo ERA5 reutiliza o leitor/interpolador contratado e confronta o resultado float32 exatamente com staging. Verifica reprodutibilidade do processamento; não é uma validação independente da geometria, do algoritmo de interpolação ou da validade meteorológica. A auditoria compara a seleção temporal com o manifesto congelado; não refaz a busca nearest2 no acervo inteiro.

Nenhuma fonte, cache, staging, contrato ou Zarr é alterado. Temporários DuckDB e PNGs decodificados ficam sob o diretório da auditoria e são removidos ao terminar normalmente. Não executar durante modificações das entradas. A execução não tem resume: após falha, preserve o relatório e use outra saída.

## Arquivos entregues

- `audit_reference_zarr.py`: auditor com flags `--all-selected-pngs` e `--recompute-era5`; mantém o modo amostral antigo se chamado sem essas flags.
- `scripts/run_reference_audit.py`: pré-verificação, execução completa, ambiente, identidade do código e código de saída persistidos.
- `tests/test_reference_integrity_full.py`: aprovação sintética completa e rejeição de fontes PNG/ERA5 alteradas.
- Este guia e `SEQUENCIA_00_18.md`: execução e dependências da migração dos diagnósticos.

## 1. Atualizar o código

Aplique o patch na máquina local, revise, faça commit e push na branch de trabalho. No CEFET, com alterações locais já salvas:

```bash
cd /home/vsantos/repositories/rionowcastdf &&
git fetch origin &&
git switch chore/reorganizar-estrutura &&
git pull --ff-only origin chore/reorganizar-estrutura
```

Se a branch já tiver sido integrada à main, use a branch integrada. Não use reset --hard. A organização de diretórios não altera os caminhos do builder ou das entradas.

## 2. Ambiente e fontes

Use o mesmo Python/ambiente da construção. Se o ambiente virtual documentado existir:

```bash
source /home/vsantos/repositories/rionowcastdf/.venv-zarr-reference/bin/activate
```

Se construiu usando Conda, ative esse ambiente em vez do comando acima. Não crie outro ambiente desnecessariamente. Os requisitos de referência estão em `requirements_reference_zarr.txt`; somente se for necessário preparar um ambiente novo, use esse arquivo. A auditoria não exige GPU, CUDA ou PyTorch.

```bash
cd /home/vsantos/repositories/rionowcastdf
export RADAR_ROOT=/home/vsantos/data/radar_sumare
export ERA5_ROOT=/home/vsantos/data/reanalysis/cds/era5
mkdir -p logs
python -m unittest discover -s tests -p 'test_reference_integrity*.py' -v
```

Confira que `configs/corrdiff_reference_zarr.json` aponta para o Zarr e os caches realmente construídos. Não edite o contrato congelado no dataset para fazer uma divergência desaparecer. Preserve `datasets/corrdiff_reference_v1/staging/`.

## 3. Pré-verificar

```bash
python scripts/run_reference_audit.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --output-dir analysis_outputs_reference/integrity_full_v1 \
  --check-only
```

`preflight_passed=true` significa somente que caminhos básicos e conclusão da construção estão presentes. Ainda não verifica integridade. Se a saída já existir, escolha `integrity_full_v2` e um nome de log correspondente.

## 4. Executar com nohup

Execute após a pré-verificação aprovada, com as variáveis e o ambiente ainda ativos:

```bash
nohup python -u scripts/run_reference_audit.py \
  --contract-config configs/corrdiff_reference_zarr.json \
  --output-dir analysis_outputs_reference/integrity_full_v1 \
  > logs/integrity_full_v1.log 2>&1 &
AUDIT_PID=$!
printf '%s\n' "$AUDIT_PID" > logs/integrity_full_v1.pid
printf 'PID: %s\n' "$AUDIT_PID"
```

```bash
tail -n 80 -f logs/integrity_full_v1.log
```

Ctrl+C encerra apenas o tail. Para consultar o processo:

```bash
ps -p "$(cat logs/integrity_full_v1.pid)" -o pid,etime,%cpu,%mem,cmd
```

A ausência do processo não comprova sucesso. Confira `execution.json` e `audit/summary.json`.

## 5. Etapas internas e resultados

1. Contratos, hashes, manifesto, UTC/local, splits e purge.
2. Checksums de staging por mês.
3. Recálculo de todas as entradas ERA5 selecionadas e comparação com staging.
4. Recálculo da normalização sobre horas únicas do treino.
5. Comparação de todos os patches de input/target/mask e metadados; hashes lógicos e contagens por split.
6. Hash de cada caminho PNG selecionado e decodificação de cada conteúdo/política para comparação com cache.

Logs de progresso identificam o estágio, mês ou split. O recálculo de normalização pode passar algum tempo sem linhas intermediárias. A leitura integral pode levar horas; o tempo dependerá do disco, dos Parquets e da carga do servidor. Não estimar a duração apenas pelo número de PNGs. Haverá uso temporário de disco pelo DuckDB sob a saída da auditoria.

Arquivos da execução:

| Arquivo | Conteúdo |
|---|---|
| `execution.json` | Comando, Python, commit, hashes, horário e exit_code |
| `environment.txt` | Dependências efetivamente instaladas |
| `audit/summary.json` | Resultado da auditoria e contagens |
| `audit/png_paths_verified.jsonl` | Todos os caminhos PNG conferidos com seus hashes |
| `audit/png_samples.json` | Conteúdos decodificados; apesar do nome herdado, no modo completo contém todos os hashes/políticas selecionados |
| `audit/era5_sources_verified.json` | Identidade dos Parquets efetivamente relidos |

Sucesso exige `execution.json`: `exit_code=0` e `integrity_passed=true`; e `audit/summary.json`: `complete=true`, `integrity_passed=true`, `all_selected_png_paths_checked=true`, `all_selected_png_contents_decoded=true`, `era5_raw_recomputed=true`.

`execution.complete=true` significa que o lançador terminou, inclusive em caso de falha; não basta para aprovação. Interrupção abrupta pode deixar `complete=false`. O auditor interrompe na primeira inconsistência e não realiza reparos. `production_ready=false` permanece porque ainda faltam diagnósticos e integração nativa.

## 6. Enviar os resultados

```bash
cat analysis_outputs_reference/integrity_full_v1/execution.json
cat analysis_outputs_reference/integrity_full_v1/audit/summary.json
tail -n 100 logs/integrity_full_v1.log
```

Envie esses dois JSONs e o final do log, mesmo em caso de erro. Não é necessário enviar PNGs, staging, cache ou Zarr. Para empacotar os relatórios pequenos após terminar:

```bash
tar -czf /home/vsantos/integrity_full_v1_reports.tar.gz \
  -C /home/vsantos/repositories/rionowcastdf \
  analysis_outputs_reference/integrity_full_v1/execution.json \
  analysis_outputs_reference/integrity_full_v1/environment.txt \
  analysis_outputs_reference/integrity_full_v1/audit/summary.json \
  logs/integrity_full_v1.log
```

Se uma falha ocorrer antes da criação do relatório, envie o log. Preserve os manifests extensos no servidor; não é necessário incluí-los no Git.
