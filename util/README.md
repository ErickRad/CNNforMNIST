# CNN para Classificação de Dígitos Manuscritos (MNIST)

Este projeto implementa uma Rede Neural Convolucional (CNN) usando PyTorch para classificar dígitos manuscritos do famoso dataset **MNIST**.

## 🧠 Arquitetura da Rede

A arquitetura foi desenvolvida para capturar padrões visuais dos dígitos com três camadas convolucionais seguidas por pooling e duas camadas totalmente conectadas.

Entrada: (1, 28, 28)

[Conv2D] -> 32 filtros 3x3 (padding=1)
[ReLU]
[MaxPool2D] 2x2

[Conv2D] -> 64 filtros 3x3 (padding=1)
[ReLU]
[MaxPool2D] 2x2

[Conv2D] -> 128 filtros 3x3 (padding=1)
[ReLU]
[MaxPool2D] 2x2

Flatten
[Linear] -> 128 neurônios
[ReLU]
[Linear] -> 10 classes (0 a 9)

## 📦 Requisitos

- Python 3.8+
- PyTorch
- torchvision
- matplotlib (opcional, para visualizar resultados)

Instale as dependências com:

```bash
pip install torch torchvision matplotlib
