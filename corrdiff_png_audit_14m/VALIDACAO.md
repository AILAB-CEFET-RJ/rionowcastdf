# Validação desta entrega

Quatro testes automatizados passaram no ambiente de preparação:

1. Transparência total; cartografia com coincidência de cor de paleta; opaco sem paleta; paleta fora da amostragem; paleta amostrada; alpha parcial excluído da paleta; 5880 pontos geométricos.
2. Arquivo ilegível, PNG truncado, dimensões 1024x768 e JPEG com extensão PNG.
3. Conversão local -> UTC com horário de verão; hora ambígua, hora inexistente e nome não interpretável.
4. Execução ponta a ponta: inventário, deduplicação por SHA, relatórios, amostras, ZIP, retomada por stat, alteração de origem, remoção de arquivo, --rehash, trava exclusiva e link simbólico sinalizado.

Dois arquivos reais previamente disponibilizados foram processados:

- 2012_02_25_04_00.png: map_screen_review, fração bege 0,36761113597374506, zero pixels de paleta.
- 2012_12_11_17_00.png: palette_sampled_observation_validity_unconfirmed, 17.578 pixels de paleta, 714 pontos amostrados na grade de referência.

O painel produzido com esses dois casos foi inspecionado visualmente: cartografia aparece como imagem opaca e o controle com ecos aparece sobre fundo quadriculado na área transparente.

Não houve execução no servidor do CEFET nem validação do acervo completo nesta entrega. Não foi certificado o significado meteorológico da transparência, a origem das telas cartográficas, a cobertura física ou a invariância da geometria entre 2011 e 2024. O algoritmo cartográfico é deliberadamente uma heurística específica do padrão bege conhecido, não um classificador universal.
