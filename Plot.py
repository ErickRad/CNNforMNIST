import matplotlib.pyplot as plt
import numpy as np
import torch

def plotar_perdas(perdas_treino, perdas_teste, epocas):
    plt.figure()
    plt.plot(range(1, epocas + 1), perdas_treino, label='Perda de Treino')
    plt.plot(range(1, epocas + 1), perdas_teste, label='Perda de Teste')
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.title("Perda de Treino e Teste")
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0) 
    plt.savefig("util/perda.png")
    plt.close()
    print("Gráfico de perda salvo como perda.png")

def plotar_classificacoes(modelo, dataloader, device):
    modelo.eval()
    imagens, rotulos = next(iter(dataloader))
    imagens, rotulos = imagens[:10].to(device), rotulos[:10]
    saidas = modelo(imagens)
    _, previsoes = torch.max(saidas, 1)
    fig, axs = plt.subplots(1, 10, figsize=(15, 3))

    for i in range(10):
        axs[i].imshow(imagens[i].cpu().squeeze(), cmap='gray')
        axs[i].set_title(f"{previsoes[i].item()}")
        axs[i].axis('off')

    plt.savefig("util/classificacoes.png")
    plt.close()
    print("Gráfico de classificações salvo como classificacoes.png")


def plotar_precisoes(precisoes):
    labels = [str(i) for i in range(10)]
    num_classes = len(labels)

    angulos = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    precisoes += precisoes[:1]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angulos, precisoes, marker='o', linestyle='-', linewidth=2, color='mediumslateblue')
    ax.fill(angulos, precisoes, color='mediumpurple', alpha=0.25)

    ax.set_thetagrids(np.degrees(angulos[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("Precisão por Dígito (Radar)", size=14, weight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig("util/precisao.png")
    plt.close()
    print("Radar de precisão salvo como precisao.png")