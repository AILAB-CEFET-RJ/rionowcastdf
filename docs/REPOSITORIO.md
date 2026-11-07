# Mapa do repositório

Organização inicial baseada na main 9d7cd6bd7a250a6a3b7c868f9b1d5517a330504e, em 07/11/2026.
A reorganização estrutural agrupa os módulos de preparação em `pipelines/dataset_preparation/`, os documentos em `docs/` e as amostras em `examples/radar_sumare/`. Contratos e evidências geradas permanecem inalterados.

## Código e documentação

| Caminho | Papel e tratamento |
|---|---|
| src/corrdiff/builder/ | Fluxo oficial de cache/Zarr no modo --contract-config. Preservar caminhos e conteúdo durante a auditoria. |
| configs/ | Configurações; corrdiff_reference_zarr.json é a entrada da referência atual. |
| audit_reference_zarr.py | Auditoria integral de tensores, staging e cache; PNGs por amostragem. |
| scripts/ | Diagnósticos 0–18; requerem adaptação ao alvo code/50, máscara e Zarrs separados. |
| adapter/ | Integração existente; ainda precisa de validação nativa com a referência. |
| tests/ | Testes automatizados do builder e da integração; testes locais não substituem auditoria do acervo. |
| notebooks/, experimental/ | Exploração e experimentos; não são o fluxo oficial. |
| docs/documents/, docs/guides/, figures/ | Documentação e figuras existentes. Não atribuir resultados antigos ao novo dataset. |
| docs/history/README_RECONSTRUCAO_FASES_00_A_18.md | Roteiro histórico preservado; não é uma certificação de compatibilidade com o novo Zarr. |

## Preparação e evidências históricas

| Caminho | Função |
|---|---|
| pipelines/dataset_preparation/corrdiff_extreme_audit/, pipelines/dataset_preparation/corrdiff_source_audit/ | Investigação de extremos e rastreabilidade das fontes. |
| pipelines/dataset_preparation/corrdiff_radar_qc/, pipelines/dataset_preparation/14g_cruzar_radar.py | Políticas e cruzamentos anteriores à confirmação do produtor. |
| pipelines/dataset_preparation/verificacao_espacial_14h/, pipelines/dataset_preparation/corrdiff_alignment_14i/ | Inspeções espaciais e temporais; evidências da correção. |
| pipelines/dataset_preparation/corrdiff_cache_14j/ | Cache intermediário decodificado e geometria de referência. |
| pipelines/dataset_preparation/corrdiff_pairing_14k/ | Inventário ERA5 e pareamento. |
| pipelines/dataset_preparation/corrdiff_smoke_14l/ | Teste técnico anterior à política final do produtor. |
| pipelines/dataset_preparation/corrdiff_png_audit_14m/ | Auditoria completa do acervo PNG. |
| pipelines/dataset_preparation/corrdiff_policy_14n/ | Política baseada nas respostas do produtor. |
| pipelines/dataset_preparation/corrdiff_contract_14o/ | Preparação do contrato espacial e numérico. |
| pipelines/dataset_preparation/corrdiff_temporal_14p/ | Seleção temporal e congelamento do manifesto. |
| examples/radar_sumare/amostras_orientador_sumare/ | Exemplos de fonte, incluindo arquivos deliberadamente inválidos para demonstração. |
| recovery_runs/, recovery_comparison/ | Recuperação de versões anteriores, fora do fluxo oficial. |

As etapas 14j–14p não são lixo: algumas produzem entradas ainda usadas e outras documentam decisões científicas. Os módulos foram movidos em conjunto, preservando o conteúdo Python e os imports locais. Consulte `docs/MIGRACAO_ESTRUTURA.md` para os caminhos anteriores.

## Caminhos protegidos nesta etapa

A configuração oficial referencia:
- data/radar_intermediate_14j_v1/
- analysis_outputs_restored/14k_pairing_v2/
- analysis_outputs_restored/14n_policy_v1/
- analysis_outputs_restored/14o_prepared_s8_v1/
- analysis_outputs_restored/14p_temporal_contract_v1/
- data/radar_reference_v1/
- datasets/corrdiff_reference_v1/

Preservar também datasets/corrdiff_reference_v1/staging/ até terminar a auditoria. Não alterar módulos contract_*.py apenas para reformatação: os hashes do código fazem parte do contrato congelado.

## Próxima limpeza física

Antes de retirar dados volumosos do Git, confirmar backup e manifesto de recuperação. Remover do índice preservando cópia local não reduz o histórico já publicado. Reescrita de histórico é uma operação distinta, coordenada com colaboradores.

As amostras PNG e o ZIP estão agrupados em `examples/radar_sumare/`; backups de código permanecem em `archive/code_snapshots/`. Nenhuma remoção em massa faz parte desta etapa.
