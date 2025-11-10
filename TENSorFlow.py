🟢 Básico (1–5)

Instale o TensorFlow e verifique sua versão.

Crie um tensor 1D, 2D e 3D e imprima suas formas (shape).

Converta uma lista Python em tensor e vice-versa.

Realize operações matemáticas básicas entre tensores (soma, multiplicação, média).

Gere tensores de zeros, uns e valores aleatórios.

🟡 Intermediário (6–10)

Crie uma função que calcule a média e o desvio-padrão de um tensor.

Monte uma rede neural densa simples para prever a saída de y = 2x + 1.

Treine a rede, calcule perda (loss) e avalie a performance.

Salve o modelo treinado em arquivo .h5 e carregue novamente.

Use o TensorBoard para visualizar o treino.

🔵 Avançado (11–15)

Implemente uma rede convolucional (CNN) para classificar imagens MNIST.

Aplique Dropout e BatchNormalization para melhorar generalização.

Crie um callback personalizado que pare o treino se a acurácia estagnar.

Plote a curva de perda e acurácia durante o treino.

Compare resultados entre SGD, Adam e RMSprop.

🔴 Sênior (16–25)

Crie uma rede LSTM para prever séries temporais (ex.: temperatura diária).

Use tf.data para carregar dados em lote (batch) de forma eficiente.

Monte um pipeline de pré-processamento com normalização e one-hot encoding.

Implemente transfer learning com o modelo MobileNetV2.

Fine-tune a rede e salve o melhor modelo automaticamente.

Treine o modelo com GPU (se disponível) e meça o tempo de execução.

Converta o modelo para TensorFlow Lite e exporte para uso em dispositivos móveis.

Use tf.function para otimizar uma função com graph execution.

Treine dois modelos em paralelo com tf.distribute.MirroredStrategy.

Faça um script que avalie métricas personalizadas (precisão, recall, F1-score) com tf.metrics.