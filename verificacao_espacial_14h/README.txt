VERIFICAÇÃO ESPACIAL 14h

Instalar dependências, se necessário:
python -m pip install numpy pillow matplotlib scipy

Na raiz do repositório no CEFET, com o script nesta pasta:
python -u 14h_verificacao_espacial.py --radar-dir /home/vsantos/data/radar_sumare --output-dir analysis_outputs_restored/14h_spatial_examples

O diretório de saída deve ser novo. Os três exemplos padrão pertencem ao treino:
2011_08_27_11_00; 2016_10_02_14_00; 2014_10_02_04_00.
Pode usar --timestamps YYYY_MM_DD_HH_MM ... para outros casos.
Aceita árvore YYYY/MM/DD ou cópias na raiz de --radar-dir.

SAÍDAS
spatial_summary.csv: totais por imagem, quantidade de cores da paleta capturadas/perdidas pela amostragem.
summary.json: parâmetros, hashes, limitações e resultados.
*_pixels.csv: coordenadas de cada pixel opaco da paleta, classe, ponto da grade mais próximo e distância em pixels do PNG.
*_components.csv: grupos contíguos (vizinhança 8), caixas delimitadoras e interseção com a amostragem.
*_panel.png: original com transparência visível, grade sobreposta e zoom do maior componente.

INTERPRETAÇÃO
palette_present_but_not_sampled: prova que a amostragem pontual não capturou cores presentes no PNG. Não prova que sejam ecos meteorológicos nem que uma agregação espacial deva substituí-la.
palette_sampled: ao menos uma cor exata opaca coincide com a grade.
no_opaque_exact_palette: ausência de correspondência na paleta adotada; não equivale a tempo seco ou falha.

A geometria reproduz o método de precompute_pixel_map do builder consultado na main (blob 4e0339320cdee0d16cd31217bfc374482fd5015d). Latitude/longitude e resolução são os parâmetros da auditoria anterior, fixados neste diagnóstico. Não são uma validação independente do georreferenciamento. Origem das coordenadas PNG: canto superior esquerdo; x=coluna, y=linha. Distâncias são pixels de origem, não km. Pixels com alpha diferente de 255 não são contados como paleta opaca.

O script não lê ou modifica o cache, não reconstrói o dataset e não abre testes para seleção de modelos. Recusa dimensões diferentes de 656x654 e recusa sobrescrever o diretório de saída. Erros individuais ficam no resumo e causam código de saída 1.

Verificação local realizada com imagem sintética: um pixel amostrado, um não amostrado e uma cor com alpha=0; contagens e saídas conferidas. Os três PNGs reais precisam ser processados onde está o acervo. Enviar spatial_summary.csv e os três painéis para interpretação.
