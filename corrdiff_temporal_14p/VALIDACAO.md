# Validação da entrega 14p

Data: 2026-11-06. Seis testes unitários/de integração passaram localmente.

- Horário de verão: horário inexistente e ambíguo rejeitados; exemplo 25/02/2012 03h convertido para 05h UTC.
- Política: transparente elegível; desconhecido sem paleta excluído; mapa com eco não aceito como seco.
- Seleção: tolerância inclusiva, empate pelo anterior, exclusão de candidato de outro split.
- Limites da purga: intervalos semiabertos de ±48h.
- Revisão: aceite exige evidência, revisor e data.
- Integração: banco 14m sintético + manifesto 14k; snapshot/política/relatórios/freeze; 1 hora exata e 2 com nearest2; hash do banco original inalterado; timestamps duplicados excluídos; delta do cache correto; saída existente rejeitada; adulteração do manifesto detectada antes de freeze.

Não houve execução nos 3.059.811 arquivos do CEFET: audit.sqlite e cache estão no servidor do usuário. O pacote utiliza o schema publicado da auditoria e do pareamento. Os resultados científicos finais dependem da execução real e da decisão temporal documentada.
