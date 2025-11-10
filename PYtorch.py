# Básico (1–5)

# Instale o PyTorch e mostre a versão instalada.

# Crie tensores 1D, 2D e 3D e exiba seus tamanhos e tipos de dados.

# Converta uma lista Python ou um numpy array em tensor PyTorch.

# Execute operações matemáticas básicas (soma, multiplicação, média, desvio padrão).

# Gere tensores aleatórios com distribuição normal e uniforme.

# 🟡 Intermediário (6–10)

# Crie uma função que normalize um tensor entre 0 e 1.

# Use o módulo torch.nn para montar uma rede neural simples com 1 camada oculta.

# Crie um loop de treino manual (sem Trainer) usando loss.backward() e optimizer.step().

# Visualize o valor da função de perda a cada época.

# Salve e carregue o modelo treinado usando torch.save() e torch.load().

# 🔵 Avançado (11–15)

# Implemente uma rede convolucional (CNN) para classificar o dataset MNIST.

# Use torchvision.datasets e DataLoader para carregar dados em lote.

# Adicione Dropout e BatchNorm à rede e meça a melhoria no desempenho.

# Plote as curvas de loss e accuracy durante o treino.

# Compare o desempenho usando diferentes otimizadores: SGD, Adam e RMSprop.

# 🔴 Sênior (16–25)

# Monte uma LSTM para prever uma série temporal (ex.: dados de vendas).

# Crie uma rede de autoencoder para reduzir dimensionalidade de dados.

# Treine uma GAN (Generative Adversarial Network) simples que gere imagens artificiais.

# Use transfer learning com ResNet18 para classificar novas imagens.

# Implemente um callback personalizado que salve checkpoints automáticos.

# Compare resultados com e sem GPU (use device = torch.device("cuda" if torch.cuda.is_available() else "cpu")).

# Aplique quantização para reduzir tamanho do modelo e medir a diferença de desempenho.

# Integre o modelo PyTorch com ONNX e exporte-o.

# Use PyTorch Lightning para reescrever o treino de forma mais limpa e modular.

# Implemente um modelo multitarefa que preveja duas saídas diferentes ao mesmo tempo.