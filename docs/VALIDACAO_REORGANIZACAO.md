# Validação da reorganização

Base: `46d81677eba332f914b212284a066d720744483f`.

Resultado: 73 testes unittest e uma integração sintética da auditoria de fontes aprovados. Suítes executadas separadamente, a partir da raiz, com os módulos nos novos caminhos.

| Suíte | Testes | Resultado |
|---|---|---|
| corrdiff_alignment_14i | 9 | aprovado |
| corrdiff_cache_14j | 6 | aprovado |
| corrdiff_contract_14o | 8 | aprovado |
| corrdiff_extreme_audit | 4 | aprovado |
| corrdiff_pairing_14k | 7 | aprovado |
| corrdiff_png_audit_14m | 4 | aprovado |
| corrdiff_policy_14n | 6 | aprovado |
| corrdiff_radar_qc | 11 | aprovado |
| corrdiff_smoke_14l | 5 | aprovado |
| corrdiff_source_audit | integração via script | aprovado |
| corrdiff_temporal_14p | 6 | aprovado |
| test_reference_zarr.py | 5 | aprovado |
| test_reference_integrity.py | 2 | aprovado |

O teste do builder foi repetido com `xarray==2024.11.0`, conforme requirements_reference_zarr.txt. A versão mais recente instalada inicialmente era incompatível com Zarr 2. Nenhuma dependência do repositório foi alterada.

Verificações adicionais: 44 arquivos Python movidos sem alteração de bytes; src/, configs/ e árvores de resultados/recuperação sem mudanças; git diff --check sem erros.

Esta validação cobre a reorganização e dados sintéticos. Não executa novamente a auditoria do Zarr completo do CEFET nem certifica integração nativa CorrDiff.
