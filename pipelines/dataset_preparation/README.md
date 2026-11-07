# Preparação do dataset de referência

Esta pasta reúne as etapas de investigação, auditoria e preparação que levaram ao Zarr. Execute os comandos **a partir da raiz do repositório**, acrescentando `pipelines/dataset_preparation/` ao caminho anterior do programa.

Exemplo de consulta à interface:

```bash
python pipelines/dataset_preparation/corrdiff_pairing_14k/pair_inventory.py --help
```

Cada etapa mantém seu README, requisitos e testes. Não é necessário executar novamente todas as etapas para usar o Zarr existente.

| Etapas | Finalidade |
|---|---|
| Auditorias de extremos, fontes, QC e 14g–14i | Investigação e alinhamento espacial e temporal |
| 14j–14k | Cache intermediário, inventário ERA5 e pareamento |
| 14l | Verificação técnica de carregamento e treinamento mínimo |
| 14m | Auditoria completa do acervo de PNGs |
| 14n | Semântica confirmada pelo produtor |
| 14o | Contrato espacial e numérico |
| 14p | Seleção temporal e manifesto |

O fluxo oficial de construção continua em `src/corrdiff/builder/`, com configuração em `configs/`. Os resultados continuam em `analysis_outputs_restored/`; caches e datasets continuam em `data/` e `datasets/`.

Consulte [a migração de caminhos](../../docs/MIGRACAO_ESTRUTURA.md) e [o mapa do repositório](../../docs/REPOSITORIO.md).
