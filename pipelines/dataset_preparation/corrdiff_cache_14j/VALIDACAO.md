# Validação da entrega 1

Data: 2026-11-02.

Seis testes automatizados passaram, abrangendo os casos descritos no README.
Três PNGs originais foram recuperados da raiz do repositório
AILAB-CEFET-RJ/rionowcastdf e processados pelo novo cache:

| PNG local | UTC | Classificação | Pontos de paleta na grade de referência |
|---|---|---|---:|
| examples/radar_sumare/2012_02_25_03_00.png | 2012-02-25 05:00Z | transparent_ambiguous | 0 |
| examples/radar_sumare/2012_02_25_04_00.png | 2012-02-25 06:00Z | map_screen_review | 0 |
| examples/radar_sumare/2012_12_11_17_00.png | 2012-12-11 19:00Z | palette_sampled_observation_validity_unconfirmed | 714 |

O último caso reproduz os 229 pontos da geometria legada e 714 da referência do
piloto 14i. A transparência não recebeu zero. Os originais foram preservados.
Não foi executado o acervo completo no CEFET; não foi realizado treinamento ou
pareamento com ERA5. A validade meteorológica e a calibração histórica da geometria
continuam dependentes da confirmação do produtor.
