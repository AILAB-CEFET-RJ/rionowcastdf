# Resultados científicos do dataset de referência

Este diretório contém **metadados pequenos e resumos versionados** das auditorias e dos diagnósticos referentes a `datasets/corrdiff_reference_v1`. Dados intermediários volumosos, Zarrs e fontes completas permanecem no armazenamento de pesquisa.

## Situação verificada em 12/11/2026

- `integrity_full_v1/audit/summary.json`: `complete=true`, `integrity_passed=true`, **99.098 horários**, 4.756.704 patches divididos em quatro conjuntos; fontes selecionadas e tensores comparados conforme flags do relatório.
- `shared_reader_v2/summary.json`: `reader_check_passed=true` nas amostras declaradas.
- `phases_v1/00_structure/summary.json` até `phases_v1/17_ablations/summary.json`: `complete=true` e `passed=true` em cada fase, **dentro do escopo específico** registrado em `scope`.
- Fase 16: baselines clássicos amostrais de treino/validação, **sem CorrDiff nativo**.
- Fase 17: **66 execuções planejadas**, `training_executed=false`.

**Limites importantes:** `production_ready=false` não foi alterado; validade física do suporte radar não foi confirmada apenas pela auditoria computacional. O target atual é `code / 50`, não `log1p`. As fases 18 (smoke nativo) e 19+ (treino) têm evidências separadas e ainda não estão concluídas.

Use os links e instruções do [README principal](../README.md), especialmente as seções de contrato, interpretação das máscaras e avaliação. Não sobrescrever relatórios de uma execução com resultados de outra identidade de dataset.
