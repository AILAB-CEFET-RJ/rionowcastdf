# Migração de caminhos

Execute os comandos a partir da raiz do repositório. Os caminhos de dados, resultados, configurações e do builder não mudaram. Os arquivos Python movidos mantêm o conteúdo original; imports entre arquivos da mesma etapa continuam locais.

| Caminho anterior | Caminho atual |
|---|---|
| `14g_cruzar_radar.py` | `pipelines/dataset_preparation/14g_cruzar_radar.py` |
| `2012_02_25_03_00.png` | `examples/radar_sumare/2012_02_25_03_00.png` |
| `2012_02_25_04_00.png` | `examples/radar_sumare/2012_02_25_04_00.png` |
| `2012_12_11_17_00.png` | `examples/radar_sumare/2012_12_11_17_00.png` |
| `amostras_orientador_sumare` | `examples/radar_sumare/amostras_orientador_sumare` |
| `corrdiff_alignment_14i` | `pipelines/dataset_preparation/corrdiff_alignment_14i` |
| `corrdiff_cache_14j` | `pipelines/dataset_preparation/corrdiff_cache_14j` |
| `corrdiff_contract_14o` | `pipelines/dataset_preparation/corrdiff_contract_14o` |
| `corrdiff_extreme_audit` | `pipelines/dataset_preparation/corrdiff_extreme_audit` |
| `corrdiff_pairing_14k` | `pipelines/dataset_preparation/corrdiff_pairing_14k` |
| `corrdiff_png_audit_14m` | `pipelines/dataset_preparation/corrdiff_png_audit_14m` |
| `corrdiff_policy_14n` | `pipelines/dataset_preparation/corrdiff_policy_14n` |
| `corrdiff_radar_qc` | `pipelines/dataset_preparation/corrdiff_radar_qc` |
| `corrdiff_smoke_14l` | `pipelines/dataset_preparation/corrdiff_smoke_14l` |
| `corrdiff_source_audit` | `pipelines/dataset_preparation/corrdiff_source_audit` |
| `corrdiff_temporal_14p` | `pipelines/dataset_preparation/corrdiff_temporal_14p` |
| `documents` | `docs/documents` |
| `guides` | `docs/guides` |
| `verificacao_espacial_14h` | `pipelines/dataset_preparation/verificacao_espacial_14h` |

Relatórios, manifests e contratos já gerados preservam seus caminhos históricos como evidência. Esta tabela permite localizar o código correspondente sem reescrever esses artefatos.

Após aplicar a migração, processos já iniciados devem terminar antes de iniciar novos comandos nos caminhos atualizados. O Zarr e o cache existentes não precisam ser reconstruídos por causa desta organização.

Os arquivos SHA256SUMS e relatórios de validação das entregas antigas registram os pacotes originais. A documentação operacional foi atualizada; esses registros históricos não são um checksum da árvore reorganizada. O conteúdo Python foi comparado byte a byte com a versão anterior.
