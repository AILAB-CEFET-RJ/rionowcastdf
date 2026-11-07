# Mapa do repositório

Organização inicial baseada na main 9d7cd6bd7a250a6a3b7c868f9b1d5517a330504e, em 07/11/2026.
Esta etapa organiza a navegação e o versionamento; não move módulos, não altera contratos e não elimina evidências.

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
| documents/, guides/, figures/ | Documentação e figuras existentes. Não atribuir resultados antigos ao novo dataset. |
| README_RECONSTRUCAO_FASES_00_A_18.md | Roteiro histórico preservado; não é uma certificação de compatibilidade com o novo Zarr. |

## Preparação e evidências históricas

| Caminho | Função |
|---|---|
| corrdiff_extreme_audit/, corrdiff_source_audit/ | Investigação de extremos e rastreabilidade das fontes. |
| corrdiff_radar_qc/, 14g_cruzar_radar.py | Políticas e cruzamentos anteriores à confirmação do produtor. |
| verificacao_espacial_14h/, corrdiff_alignment_14i/ | Inspeções espaciais e temporais; evidências da correção. |
| corrdiff_cache_14j/ | Cache intermediário decodificado e geometria de referência. |
| corrdiff_pairing_14k/ | Inventário ERA5 e pareamento. |
| corrdiff_smoke_14l/ | Teste técnico anterior à política final do produtor. |
| corrdiff_png_audit_14m/ | Auditoria completa do acervo PNG. |
| corrdiff_policy_14n/ | Política baseada nas respostas do produtor. |
| corrdiff_contract_14o/ | Preparação do contrato espacial e numérico. |
| corrdiff_temporal_14p/ | Seleção temporal e congelamento do manifesto. |
| amostras_orientador_sumare/ | Exemplos de fonte, incluindo arquivos deliberadamente inválidos para demonstração. |
| recovery_runs/, recovery_comparison/ | Recuperação de versões anteriores, fora do fluxo oficial. |

As etapas 14j–14p não são lixo: algumas produzem entradas ainda usadas e outras documentam decisões científicas. Mover esses módulos requer revisão de imports, caminhos e instruções.

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

PNG soltos, ZIPs de exemplos e backups de código podem ser consolidados depois de verificar referências. Nenhuma remoção em massa faz parte desta etapa.
