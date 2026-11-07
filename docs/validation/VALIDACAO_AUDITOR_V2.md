# Validação sintética do auditor v2

Seis testes aprovados:

1. Dataset consistente e rejeição de patch alterado.
2. Recusa de saída dentro do dataset.
3. Rejeição de Parquet alterado em relação à identidade congelada.
4. Verificação completa das seis aquisições PNG do fixture, incluindo conteúdos repetidos, e rejeição de PNG alterado.
5. Lançador: pré-verificação sem criar saída, execução completa, registro de sucesso e recusa de reutilização.
6. Recálculo ERA5 rejeita divergência de valores em staging mesmo quando o checksum local de staging é atualizado.

Comando: `python -m unittest discover -s tests -p 'test_reference_integrity*.py' -v`.

Também verificados: compilação Python, interfaces --help e aplicação do patch contra a revisão de base. Os módulos do builder e configurações congeladas não foram alterados.

Esses testes não são a auditoria do dataset real do CEFET. O programa não comprova a validade física da cobertura nem substitui o treinamento nativo CorrDiff. As migrações das fases científicas 0–18 permanecem explicitamente pendentes no roteiro.
