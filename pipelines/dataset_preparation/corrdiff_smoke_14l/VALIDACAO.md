# Validação local — 02/11/2026

Cinco testes automatizados passaram:

1. Interpolação bilinear reproduz campo linear conhecido; extrapolação e coordenadas duplicadas são recusadas.
2. Loss e gradientes permanecem iguais quando valores fora do suporte mudam, inclusive com NaN; gradientes das saídas excluídas são zero; máscara vazia é recusada.
3. Grade70×84 gera12posições32×32/stride16; patches sem suporte são excluídos.
4. Teste e horários ambíguos não são selecionados; falsificação de split para um timestamp2023 é recusada.
5. Pipeline sintético completo: inventário14k, pareamento, leitura ERA5+NPZ radar, interpolação, normalização, patches e3passosCPU. Pesos atualizados, loss/gradientes finitos, NaNs preservados no target bruto. NPZ corrompido posteriormente é recusado por checksum.

Ambiente de teste: PyTorch2.15.0+cpu, SciPy1.17.0, DuckDB2.0.0, PyArrow26.0.0.
Interface CLI --help também verificada. Não houve execução no servidor CEFET nem
treinamento CorrDiff nativo. Dados sintéticos verificam o contrato de implementação,
não a interpretação meteorológica dos PNGs históricos.
