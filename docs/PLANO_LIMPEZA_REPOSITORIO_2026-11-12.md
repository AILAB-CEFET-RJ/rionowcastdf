# Auditoria e plano de limpeza do repositório — 12/11/2026

**Repositório:** `AILAB-CEFET-RJ/rionowcastdf`  
**Branch inspecionada:** `chore/reorganizar-estrutura`  
**Commit-base:** `7029e758e4366c00fcbb7e7f0e1457af50cb581c`  
**Natureza:** inventário remoto pelo GitHub Git Trees API e leitura de contratos/relatórios. Não houve varredura dos arquivos locais não versionados no servidor CEFET. Tamanhos são aproximações decimais dos blobs da árvore do commit, **não** uso real de disco, peso total do histórico ou volume após compressão do Git.

## 1. Diagnóstico geral

O repositório já tem uma boa separação de código e histórico, mas contém artefatos gerados, dados exploratórios e backups rastreados. A regra `.gitignore` **não** retira do índice arquivos que já foram commitados.

| Árvore rastreada | Blobs aproximados | Tamanho dos blobs | Avaliação |
|---|---:|---:|---|
| `experimental/` | 5.524 | **1.041 MB** | Principal candidata a externalização; grande volume em `donwscaling/data/` |
| `analysis_outputs_restored/` | 663 | **297 MB** | Misto: evidências históricas e **entradas ativas** do contrato; não excluir em lote |
| `exports/` | 107 | **152 MB** | ZIP e extração do Capítulo 5 parcialmente duplicados |
| `analysis_outputs/` | 216 | **123 MB** | Resultados históricos; preservar resumos e arquivar binários |
| `recovery_comparison/` | 68 | **70 MB** | Inclui um log versionado de 63,36 MB |
| `figures/` | 93 | 45 MB | Figuras científicas, preservar seletivamente |
| `notebooks/` | 33 | 10 MB | Exploração; pode conter saídas embutidas |
| `docs/` | 47 | 9 MB | Documentação e evidências, manter |
| `analysis_outputs_reference/` | 23 | menos de 1 MB | Resumos atuais, **manter** |

A soma não equivale ao tamanho do repositório Git: objetos antigos, packfiles e overhead são contabilizados separadamente.

## 2. Achados prioritários de correção — antes dos smoke tests

1. **Contrato divergente:** `configs/phase18_contract.json` usa alvo `radar_reflectivity_log1p`, `stored_space=log1p(clipped radar dBZ)` e commit PhysicsNeMo `beafeecbd01fcac3c76bc6f3391a6af1872ea6bf`. A referência auditada usa `code/50` e o Doctor espera `c15bd23eb56216c32bd3e30180593e5685be9bc2`. **Não alterar só o JSON**: corrigir contrato, consumidores, testes, identificação de dataset e geração de configuração Hydra como uma mudança integrada.
2. **Leitor de gerações diferentes:** `scripts/18_prepare_conditioning_stats.py` ainda abre `datasets/corrdiff_2011_2024` e declara saída `log1p`. `adapter/reference_corrdiff_dataset.py` lê a referência correta, mas **não** implementa a interface de tupla nativa anterior nem C07 derivado. O smoke C00/C07 deve ser impedido até a reconciliação.
3. **Requisitos de ambiente antigos:** `requirements_phase18.txt` restringe `zarr<3`, enquanto a instalação aprovada `corrdiff_reference_reproduced_cu129_v4` usa Zarr 3.1.6. Arquivar/renomear como legado ou atualizar requisitos por finalidade, sem tocar no venv congelado.
4. **README principal desatualizado:** afirmava que auditoria integral e diagnósticos de referência estavam pendentes, mas `analysis_outputs_reference/integrity_full_v1/audit/summary.json` reporta PASS e `phases_v1/00...17` têm `complete=true`/`passed=true` no escopo declarado.
5. **README da Fase 18 transitório:** descreve instalação a partir do ambiente v3, já apagado no CEFET, e inclui avisos anteriores à aprovação do Doctor. Substituir por registro estável de execução e reconstrução via bundle v6.

Essas cinco correções têm impacto de **correção científica e execução**, maior do que a simples economia de espaço. O material preparado neste pacote atualiza os READMEs e documenta as divergências, mas **não** muda contrato e entrypoints de treinamento sem testes de compatibilidade.

## 3. Candidatos à remoção direta do código versionado

**Risco baixo após inspeção de `git grep` e `git status`:**

| Caminho | Motivo | Ação proposta |
|---|---|---|
| `.corrdiff_builder_2011_2024.log.swp` (0 B) | Arquivo temporário do editor | `git rm` |
| `scripts/reference_environment.py.backup_pre_v6` | Backup da implementação durante testes de empacotamento | `git rm` após preservar commit histórico |
| `scripts/reference_environment.py.backup_v6` | Idem | `git rm` |
| `scripts/reference_environment.py.backup_v6_1` | Idem | `git rm` |
| `scripts/reference_environment.py.backup_v6_2` | Idem | `git rm` |
| `Untitled.ipynb` (72 B) | Notebook sem nome, aparentemente vazio | Confirmar conteúdo e remover |
| `download_pressure.txt` (204 B) | Anotação/transiente de download | Confirmar uso; remover ou mover a nota para `docs/history/` |
| `corrdiff_builder.log` (~0,64 MB) | Log de execução na raiz | Retirar do índice, manter se houver valor histórico |
| `radar_cache_builder.log` (~1,84 MB) | Log de execução na raiz | Retirar do índice, manter se houver valor histórico |

Não excluir `scripts/reference_environment.py`, `scripts/reference_warp_smoke.py`, `tests/test_reference_environment.py` nem arquivos de contrato do builder.

## 4. Candidatos para externalizar após backup

### 4.1 Dados experimentais históricos

`experimental/donwscaling/data/` concentra o principal volume. A grafia `donwscaling` está no caminho atual; **não renomear indiscriminadamente** porque notebooks ou imports podem referenciá-lo.

- Proposta: guardar cópia com manifesto SHA-256 em área externa, por exemplo `~/data/archived_experimental/donwscaling/`, validar `rsync`/checksums e retirar o diretório **do índice Git**.
- Após validação, deixar no Git `experimental/README.md` com descrição, origem, licenças de dados, hash de manifesto e procedimento de restauração.
- Cuidado: `git rm -r --cached -- experimental/donwscaling/data/` **não apaga os arquivos do disco de trabalho**, mas os exclui de futuros commits; ainda permanecerão no histórico antigo.

### 4.2 Exportações duplicadas

Em `exports/` estão simultaneamente `capitulo5_evidencias_v1.zip` (~53,14 MB), árvores extraídas em `exports/capitulo5_evidencias_v1/` e arquivos grandes como `hour_metrics.csv` (~47,42 MB), além do mesmo conteúdo em Parquet (~20,78 MB).

- Escolher **um artefato de intercâmbio canônico** (ZIP versionado por SHA256 em armazenamento externo ou release) e pequenos resumos no Git.
- Não remover ZIP e pasta extraída antes de verificar que há cópia recuperável para a redação do Capítulo 5.
- Se esses arquivos servem como canal de acesso de outros colaboradores/assistentes, oferecer um novo local acessível antes da retirada.

### 4.3 Resultados históricos e recuperação

- `analysis_outputs/` (~123 MB): versões anteriores do dataset; separar `summary*.json`, tabelas pequenas/figuras úteis dos arrays `.npz`, `.parquet` intermediários.
- `analysis_outputs_restored/` (~297 MB): **não limpar em massa**. Exemplos de arquivos grandes: `14m_full_png_audit_v1/temporal_runs.csv` (~79,8 MB), `03_joint/relationship_samples.npz` (~22,5 MB), `06_diurnal/predictor_timestamp_means.parquet` (~18 MB). Alguns artefatos podem sustentar os contratos, a auditoria e os Capítulos 4/5.
- `recovery_comparison/recover_cache_20261025_184436.log` (~63,36 MB): conservar externamente/comprimir se houver valor diagnóstico, retirar do Git após confirmar que sumários e checksums de recuperação estão preservados.
- `recovery_comparison/` e `recovery_runs/`: manter scripts, metadados e relatórios de reconstrução; não apagar indiscriminadamente.

### 4.4 Notebooks e figuras

Preservar originais de figuras usadas na dissertação e notebooks que fundamentam o método. Para notebooks pesados, considerar remover **apenas outputs embutidos**, após preservar PDF/imagem relevante e verificar a execução. Limpeza de outputs altera hashes dos notebooks, mas não os resultados já registrados em evidências externas.

## 5. Arquivos e áreas protegidos

**Não modificar/descartar nesta limpeza:**

- `src/corrdiff/builder/contract_*.py` e código do builder cuja identidade está no contrato congelado;
- `configs/corrdiff_reference_zarr.json`, hashes, normalização, contratos e manifestos vigentes;
- `analysis_outputs_restored/14k_pairing_v2/`, `14n_policy_v1/`, `14o_prepared_s8_v1/`, `14p_temporal_contract_v1/` e `data/radar_intermediate_14j_v1/` sem análise dependencial e backup;
- `datasets/corrdiff_reference_v1/` e `data/radar_reference_v1/` no servidor;
- `analysis_outputs_reference/integrity_full_v1/` e `phases_v1/` (pequenos resumos de auditoria atual);
- `corrdiff_environment_bundles/reference_cu129_v6/`, `corrdiff_reference_reproduced_cu129_v4/`, relatórios de Doctor e instalação;
- testes, `adapter/`, matrizes de experimento, amostras radar selecionadas e documentação de proveniência.

## 6. Procedimento seguro — revisão primeiro

No **checkout local** correto, não execute a limpeza em um diretório de outra branch ou com arquivos não commitados.

```bash
cd "$HOME/repositories/rionowcastdf"
git branch --show-current
git status --short
git rev-parse HEAD

# Verificar referências aos quatro backups e arquivos transitórios
git grep -n -E 'reference_environment\.py\.backup_|Untitled\.ipynb|download_pressure\.txt' -- ':!docs/PLANO_LIMPEZA_REPOSITORIO_2026-11-12.md' || true

# Mostrar apenas arquivos versionados nos locais candidatos
git ls-files -- \
  scripts/reference_environment.py.backup_* \
  .corrdiff_builder_2011_2024.log.swp \
  Untitled.ipynb download_pressure.txt \
  corrdiff_builder.log radar_cache_builder.log \
  experimental/donwscaling/data exports recovery_comparison
```

Antes de qualquer alteração substancial de artefatos: fazer backup incremental com `rsync -a`, inventário `find ... -type f -print0 | sort -z | xargs -0 sha256sum`, e testar a restauração em diretório separado. Para dados com milhares de arquivos, preferir um manifesto de checksums preservado fora da árvore que será limpa.

### Proposta de limpeza mínima do índice (somente depois de revisar o diff)

```bash
# 1) Remover backups e lixo de editor do checkout e do índice
# Execute após verificar que não há dependências nem alterações locais a preservar.
git rm -- \
  .corrdiff_builder_2011_2024.log.swp \
  scripts/reference_environment.py.backup_pre_v6 \
  scripts/reference_environment.py.backup_v6 \
  scripts/reference_environment.py.backup_v6_1 \
  scripts/reference_environment.py.backup_v6_2

# 2) Logs podem sair SOMENTE DO ÍNDICE (continuam no disco local)
git rm --cached -- corrdiff_builder.log radar_cache_builder.log

# 3) Dados brutos históricos: ação opcional, só após backup testado
# git rm -r --cached -- experimental/donwscaling/data/

# 4) Conferir antes do commit
git diff --cached --stat
git diff --cached --check
```

Os itens 1–2 são exemplos de **comandos para execução deliberada pelo mantenedor**, e **não** fazem parte de um script automático desta entrega. Confirmar que a branch local está alinhada ao remoto; preservar cópia dos arquivos ainda importantes.

### `.gitignore` recomendado

As regras existentes já ignoram `/experimental/donwscaling/data/`, logs e vários arquivos grandes. Complementar, quando for apropriado:

```gitignore
# Backups transitórios de desenvolvimento
/scripts/reference_environment.py.backup_*

# ZIPs/exportações volumosas devem ficar fora do Git, após preservar cópia externa
/exports/**/*.zip

# Relatórios binários locais do novo ciclo experimental
/checkpoints/
/runs/
/tensorboard/
```

**Nota:** `gitignore` não reverte rastreamento anterior. Não aplicar padrões que ocultem contratos pequenos e resumos necessários à auditoria.

## 7. Ordem recomendada de trabalho

1. Salvar e commitar os READMEs atualizados (não mexer ainda no dataset).
2. Consolidar o script atual `reference_environment.py`; eliminar os quatro backups versionados.
3. Externalizar o **1,04 GB histórico** em `experimental/` com backup verificado e manifesto SHA-256.
4. Revisar as duplicações `exports/`, preservando acesso ao conjunto de evidências do Capítulo 5.
5. Reduzir logs e binários históricos de `recovery_comparison/` e `analysis_outputs/`.
6. Revisar **por dependência** os arquivos grandes em `analysis_outputs_restored/`, mantendo os contratos ativos.
7. Corrigir o contrato F18 e seus consumidores; executar testes de compatibilidade antes do smoke C00/C07.

**Importante:** uma remoção no índice diminui o *checkout* de novos clones, mas não necessariamente o tamanho de `.git` nos clones existentes nem o armazenamento do servidor GitHub. Reescrever histórico seria outra operação, de risco elevado para a pesquisa, hashes de contratos e clones de colaboradores, e **não é recomendada nesta etapa**.
