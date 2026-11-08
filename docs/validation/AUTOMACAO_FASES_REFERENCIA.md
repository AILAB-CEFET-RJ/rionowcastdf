# Execução selecionável das fases 0–18 no Zarr de referência

Entrega: reference-phases-v1. Base: 19db3f8 (`chore/reorganizar-estrutura`).

## Escopo e limites

O executor usa novas rotinas em `src/corrdiff/diagnostics/reference_phases/` e o
leitor compartilhado aprovado. Não chama os scripts históricos que pressupõem
log1p, 12 patches/hora ou máscara totalmente válida. Não altera fontes, cache,
Zarr, builder, normalização congelada ou conjuntos existentes.

As fases 0–17 têm rotinas executáveis no contrato atual, com o escopo abaixo.
Esta entrega não reproduz automaticamente todas as tabelas, gráficos e métodos
adicionais dos relatórios históricos: é uma nova versão de diagnóstico, com
população espacial sem duplicação de patches. Não misture suas saídas com as antigas.
A fase 18 fornece preflight e um ponto de execução externo configurável; a
adaptação do treinamento nativo com loss mascarada ainda não está implementada
por este pacote. Sem essa integração, a fase 18 fica **blocked**, com exit code 3.
Isso não invalida fases anteriores e nunca é apresentado como treinamento aprovado.

O campo passed dos diagnósticos indica conclusão das verificações implementadas,
não aprovação meteorológica da cobertura, suficiência estatística ou produção.
`production_ready` permanece false em todos os relatórios.

## Instalação

Aplique o patch na raiz do repositório. Use o ambiente `dataset` que executou a
construção e a auditoria. NumPy, pandas, pyarrow, Zarr e SciPy já são dependências
do builder. Para as fases 13 e seguintes, confira também scikit-learn:

```bash
python -c 'import numpy, pandas, pyarrow, zarr, scipy, sklearn; print("Dependências OK")'
# Somente se scikit-learn estiver ausente:
python -m pip install 'scikit-learn>=1.3,<2'
```

Em uma máquina nova, `requirements_reference_phases.txt` inclui os requisitos
versionados do builder e os requisitos analíticos. Não substitua o ambiente
CUDA de treinamento por esse ambiente de diagnósticos.

```bash
python -m unittest discover -s tests -p test_shared_reference_reader.py -v
python -m unittest discover -s tests -p test_reference_phases.py -v
```

Foram aprovados 9 testes do leitor e 6 testes da automação. O teste integrado
constrói um Zarr sintético com o builder real e executa as fases 0–17. Também
verifica retomada, alteração de configuração, metadados de fontes, limites do
purge e bloqueio da fase 18. Não houve execução destas novas fases no acervo
completo do CEFET nem treinamento GPU durante a implementação.

## Configuração

Arquivo: `configs/reference_phases.json`.

Os caminhos são relativos a project_root (por padrão, a raiz do repositório,
obtida a partir da localização do JSON). `--output-dir` explícito é relativo ao
terminal. A configuração aponta para:

- `datasets/corrdiff_reference_v1`;
- `analysis_outputs_reference/integrity_full_v1/audit/summary.json`;
- `analysis_outputs_reference/shared_reader_v2/summary.json`;
- saídas novas em `analysis_outputs_reference/phases_v1`.

Confirme os dois caminhos de evidência. As flags integrais, contagens e hashes de
metadados são verificados antes de executar. A automação não refaz o hash de cada
chunk do Zarr: pressupõe a imutabilidade do dataset já auditado. A ligação com os
relatórios anteriores usa suas contagens; isso não equivale a autenticar uma
nova cópia apenas por ter o mesmo tamanho.

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| phase0_max_hours_per_split | 0 | Zero = todas as horas. Valor positivo limita a varredura e deve ser tratado como piloto. |
| diagnostic_hours_per_split | 512 | Amostra aleatória de horas únicas, sem reposição, para diagnósticos espaciais/distribucionais custosos. Zero = todas. |
| sample_pixels_per_hour | 256 | Pontos por hora amostrada nas fases 1 e no ajuste da 16. |
| lags_hours | 1,3,6,12,24 | Defasagens exatas; gaps não são preenchidos. |
| baseline_train_hours | 512 | Horas amostradas exclusivamente no treino para ajustar ridge. |
| baseline_eval_hours | 128 | Horas de avaliação por conjunto train/validation. |
| regime_clusters | 5 | KMeans ajustado exclusivamente no treino. |
| regime_fit_max_hours | 20000 | Limite de horas para ajuste dos centroides; padronização usa treino. |
| seed | 42 | Amostras reproduzíveis. |
| experiment_seeds | 42,43,44 | Expansão do planejamento de ablações. |

Não há estimativa de chuva em mm/h. Os cálculos com zero usam códigos numéricos:
0 = ausência de eco exibido; 20–50 = classes exibidas em dBZ. Médias, variâncias
ou regressões desses códigos não são médias de refletividade linear Z.

## Seleção e dependências

```bash
python scripts/run_reference_phases.py --list
python scripts/run_reference_phases.py --phases 0,15 --plan
python scripts/run_reference_phases.py --phases 1-4 --plan
python scripts/run_reference_phases.py --phases all --plan
```

A fase 15 agora verifica os splits já congelados, em vez de criá-los novamente.
Por isso pode e deve vir antes da 0. A seleção `0,15` resulta em `15 -> 0`.
Dependências são acrescentadas automaticamente. `--no-deps` impede a inclusão,
mas continua exigindo resultados válidos das dependências no mesmo diretório.

## Executar agora: fases 0 e 15

No CEFET:

```bash
cd /home/vsantos/repositories/rionowcastdf
conda activate dataset
mkdir -p logs

nohup python -u scripts/run_reference_phases.py \
  --config configs/reference_phases.json \
  --phases 0,15 \
  > logs/reference_phases_00_15_v1.log 2>&1 &
echo "PID: $!"
tail -f logs/reference_phases_00_15_v1.log
```

A fase 15 lê todos os metadados e manifesto. A fase 0 lê/reconstrói todas as horas
por padrão, portanto é uma varredura potencialmente longa, sem GPU. O progresso
é impresso a cada 500 horas. Não reconstrói o Zarr e não relê PNGs ou ERA5 brutos.
A memória não cresce com todos os tensores; índices e a tabela de métricas
horárias ficam em memória, e os campos são processados uma hora por vez.

Resultados para compartilhar:

```bash
cat analysis_outputs_reference/phases_v1/run_summary.json
cat analysis_outputs_reference/phases_v1/15_split_audit/summary.json
cat analysis_outputs_reference/phases_v1/00_structure/summary.json
cat analysis_outputs_reference/phases_v1/15_split_audit/split_counts.csv
cat analysis_outputs_reference/phases_v1/15_split_audit/cross_split_source_distance.csv
cat analysis_outputs_reference/phases_v1/00_structure/target_counts.csv
cat analysis_outputs_reference/phases_v1/00_structure/train_normalization_comparison.csv
```

## Continuar escolhendo blocos

Use sempre a mesma configuração para aproveitar fases completas:

```bash
nohup python -u scripts/run_reference_phases.py --phases 1-4 --resume \
  > logs/reference_phases_01_04_v1.log 2>&1 &

# Execute os próximos comandos depois de o bloco anterior terminar:
python -u scripts/run_reference_phases.py --phases 5-8 --resume
python -u scripts/run_reference_phases.py --phases 9-11 --resume
python -u scripts/run_reference_phases.py --phases 12-14 --resume
python -u scripts/run_reference_phases.py --phases 16,17 --resume
```

Também é possível executar `--phases 0-17` em uma chamada, ou qualquer lista como
`--phases 4,6,9`. Não execute dois processos no mesmo diretório: há lock exclusivo.
`--resume` verifica hashes das saídas antes de reutilizá-las. Mudança de código,
configuração, evidências ou metadados exige um novo `--output-dir`.

Após falha ou interrupção, revise o log. Para repetir uma fase incompleta:

```bash
python -u scripts/run_reference_phases.py --phases 0,15 --resume --retry-failed
```

A tentativa anterior é arquivada em `failed_attempts/`, não apagada. Retomada é
por fase concluída: não há checkpoint interno de cada hora da fase 0. Se ela for
interrompida, será reexecutada do início.

## Escopo implementado por fase

| Fase | População / método | Principais artefatos |
|---|---|---|
| 0 | Todas as horas; grade reconstruída; máscara e classes; momentos físicos por ponto único | hour_metrics.parquet, input_statistics.csv, target_counts.csv, monthly_coverage.csv, train_normalization_comparison.csv, contagens espaciais NPZ |
| 1 | Horas/pontos de treino amostrados; quantis, assimetria, curtose; categorias do alvo | input_distributions.csv, target_distribution.csv, sampled_target_classes.csv |
| 2 | Oito derivados físicos por ponto; sem clipping | derived_statistics.csv, definitions.json |
| 3 | Médias horárias de treino; Pearson/Spearman com área de eco e códigos do alvo | hourly_relationships.csv |
| 4 | Limiares 20–50 dBZ; taxas por pixel/hora, SMD de extremos no treino | extreme_rates.csv, train_extreme_smd.csv |
| 5 | DJF/MAM/JJA/SON, por split, condicionado à disponibilidade | calendar_metrics.csv |
| 6 | Hora UTC e America/Sao_Paulo com horário de verão histórico | calendar_metrics.csv |
| 7 | Junção temporal exata por split; correlação, phi e razão de risco | lag_metrics.csv |
| 8 | Persistência em train/validation; interseção das máscaras; MAE/MSE e contingência | persistence_metrics.csv |
| 9 | Campos completos amostrados; semivariograma direcional, componentes de 4 vizinhos e bordas | semivariogram.csv, morphology.csv |
| 10 | Correlação temporal ERA5/alvo por posição, com suporte pareado | pixelwise_correlations.npz |
| 11 | Blocos não sobrepostos, menores nas bordas, médias ponderadas por suporte | scale_variance.csv |
| 12 | PCA das médias horárias; ajuste exclusivo train | pca_variance.csv, pca_loadings.csv, pca_train_fit.npz |
| 13 | KMeans de médias horárias; ajuste train e atribuição aos demais splits | regime_assignments.parquet, train_regime_fit.npz |
| 14 | Descrição de shift; SMD relativo ao treino, estatística KS e composição de regimes | predictor_shift.csv, regime_composition.csv |
| 15 | Todos os índices/manifesto; purge 48h, offsets ±120s, identidade de fontes e posições | hour_identity.parquet, split_counts.csv, shared_content_hashes.csv, cross_split_source_distance.csv |
| 16 | Baselines zero, média de treino e ridge pixelwise; avaliação train/validation | baseline_metrics.csv, ridge_train_fit.npz |
| 17 | C00–C10 originais, 3 seeds, regressão e regressão+difusão | experiment_manifest.json, 66 planos, sem treino |
| 18 | Preflight e execução de integração nativa externa explicitamente configurada | native_preflight.json; blocked enquanto não configurada |

As fases 5/6/14 são descrições condicionadas aos dados disponíveis. Os testes
podem ser descritos para auditoria de distribuição; não são usados para ajustar
PCA, centroides, normalização, baselines ou selecionar hiperparâmetros. As fases
3/7 não fornecem inferência causal; não há intervalos de confiança por bootstrap
nesta versão. A fase 11 não implementa FSS ou espectros. A fase 16 não substitui
um baseline espacial neural. Essas extensões e gráficos são trabalhos explícitos,
não funcionalidades ocultas do executor.

## Fase 18: contrato da integração externa

O pacote não chama os scripts antigos de treinamento/geração. Na configuração,
`native.corrdiff_dir` aponta para o exemplo que contém train.py;
`expected_commit` fixa o SHA; `masked_loss_verified` exige uma integração cuja
loss de regressão e difusão respeite máscaras. O comando deve ser uma lista de
argumentos (sem shell), com placeholders `{dataset_dir}`, `{output_dir}` e
`{experiment_manifest}`. Não invente um comando antes de adaptar o código nativo.

```bash
python -u scripts/run_reference_phases.py --phases 18 --resume
```

Sem integração configurada, retorna código 3 e relatório blocked. As fases 15/17
podem ser reutilizadas. Configure uma nova versão de saída após implementar a
integração. O processo externo deve gravar `native_result.json` contendo:

- complete, native_regression_tested, native_diffusion_tested;
- masked_loss_tested, checkpoint_reload_tested, all_steps_finite;
- parameters_changed, test_splits_untouched;
- contract_sha256 (SHA256 dos bytes do contract.json);
- target_transform = "target_code / 50".

As flags devem ser true e apoiadas pelos logs/checkpoints da integração. O
executor valida o formato e a identidade, mas não verifica independentemente os
cálculos do processo externo. Não chama isto de certificação do treinamento.

## Rastreabilidade e publicação

Cada fase grava summary.json, execution.log e hashes dos artefatos. run_config.json
registra código, configuração, metadados e evidências. run_summary.json descreve
a última seleção executada; os summaries individuais preservam fases anteriores.
Os testes de retomada impedem reaproveitar resultados com configuração alterada.

Envie ao Git somente código, configuração e relatórios pequenos selecionados.
Não envie tabelas horárias completas, amostras NPZ ou modelos sem verificar seu
tamanho. No capítulo 4, registre construção e integridade já verificadas; no
capítulo 5, use resultados destas análises somente após execução e revisão.
