# RioNowCastDF — ERA5 e radar do Sumaré

Construção e avaliação de um dataset para downscaling meteorológico com CorrDiff.

O código das análises e etapas 14g–14p está em [pipelines/dataset_preparation/](pipelines/dataset_preparation/README.md). Consulte o [mapa de migração](docs/MIGRACAO_ESTRUTURA.md) para atualizar comandos antigos.

## Fluxo de referência atual

1. Preparar contratos e manifesto temporal: etapas 14j–14p, catalogadas em [docs/REPOSITORIO.md](docs/REPOSITORIO.md).
2. Construir cache e Zarr com os scripts existentes e **--contract-config**: [guia de construção](README_ZARR_REFERENCIA.md).
3. Executar a [auditoria de integridade](README_INTEGRIDADE_REFERENCIA.md).
4. Adaptar e reexecutar os diagnósticos 0–18 para o contrato atual; depois validar a integração nativa com CorrDiff.

**Não usar o modo legado do builder para reconstruir a referência.** Os diagnósticos antigos não devem interpretar o alvo atual como log1p.

## Referência construída

Resultados informados em 07/11/2026; a auditoria independente de integridade ainda não está concluída neste registro.

| Propriedade | Valor |
|---|---|
| Configuração | [configs/corrdiff_reference_zarr.json](configs/corrdiff_reference_zarr.json) |
| Dataset local | datasets/corrdiff_reference_v1 |
| Cache local | data/radar_reference_v1 |
| Horas selecionadas | 99.098 |
| Patches | 4.756.704 |
| Grade | 70 × 84; passo angular 2/111 grau |
| Patches | 32 × 32, passo 8, bordas ancoradas |
| Entrada | 12 canais ERA5; armazenados em unidades físicas |
| Normalização de entrada | Somente treino, por hora e ponto únicos, antes dos patches |
| Alvo | target_code / 50; sem eco codificado como zero |
| Máscara | Obrigatória; zero fora do suporte é placeholder |
| Tempo | nearest2, sem agregação; ERA5 UTC, nomes PNG em America/Sao_Paulo |

Zero é ausência de eco exibido, não uma medição de 0 dBZ nem garantia de ausência de chuva. A cobertura permanece uma hipótese geométrica explícita. Construção concluída não equivale a aprovação científica nem a teste nativo do CorrDiff.

## Navegação

- [Mapa do repositório e diretórios protegidos](docs/REPOSITORIO.md)
- [Política de dados, resultados e histórico](docs/DADOS_E_RESULTADOS.md)
- [Construção e portabilidade](README_ZARR_REFERENCIA.md)
- [Auditoria de integridade](README_INTEGRIDADE_REFERENCIA.md)
- [Roteiro histórico das fases 0–18](README_RECONSTRUCAO_FASES_00_A_18.md) — comandos e premissas precisam ser revisados para esta referência.
- [Código do builder](src/corrdiff/builder/)
- [Testes](tests/)
- [Resultados da nova referência](analysis_outputs_reference/README.md)

Os dados completos não são distribuídos pelo Git. Uma clonagem não contém necessariamente os manifestos, inventários e caches necessários. Consulte as entradas obrigatórias no guia de construção.
