# Política de dados e resultados

## Versionar
Código, testes, configurações, documentação, contratos pequenos, resumos de execução, normalização, hashes e amostras selecionadas que sustentem a análise.

## Manter no armazenamento de dados
PNGs do acervo, Parquets ERA5, caches, bancos SQLite de inventário, Zarrs, staging, checkpoints e resultados intermediários volumosos. Registrar localização, versão e checksum dos artefatos indispensáveis à reprodução; o GitHub sozinho não substitui esses arquivos.

## Gerações de resultados
- analysis_outputs/: resultados históricos.
- analysis_outputs_restored/: reconstrução e auditorias; contém também contratos ainda ativos. Não mover nem limpar indiscriminadamente.
- analysis_outputs_reference/: novos diagnósticos do Zarr de referência. O diretório é uma convenção de destino; os diagnósticos ainda precisam ser adaptados.
- recovery_runs/ e recovery_comparison/: evidências de recuperação.
- experimental/donwscaling/data/: dados experimentais volumosos; preservar até confirmar backup externo.

Cada nova execução deve usar saída própria, registrar contrato, identidade do dataset e parâmetros. Não sobrescrever relatórios antigos com resultados de outro contrato. A auditoria já entregue pode continuar em analysis_outputs_restored/reference_integrity_v1 para preservar comandos existentes.

## Limitações do gitignore
Regras novas impedem a inclusão automática de arquivos não rastreados. Não retiram arquivos rastreados nem apagam versões antigas. Esta organização não reduz o histórico e não remove dados locais.

No diretório novo analysis_outputs_reference, publicar por padrão apenas README, summary.json, normalization.json e registros *.complete.json. Outros relatórios pequenos podem ser adicionados explicitamente após revisão. Não usar git add -f indiscriminadamente em diretórios de dados.

## Reprodutibilidade
Preservar configuração, versões das dependências, manifestos completos e checksums das fontes. Resumos pequenos permitem revisar uma execução, mas não substituem os manifestos e inventários exigidos pelo builder.
