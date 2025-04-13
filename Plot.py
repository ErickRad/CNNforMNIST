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
    imagens_coletadas = {i: [] for i in range(10)}
    rotulos_previstos = {i: [] for i in range(10)}

    with torch.no_grad():
        for imagens, _ in dataloader:
            imagens = imagens.to(device)
            saidas = modelo(imagens)
            _, previsoes = torch.max(saidas, 1)

            for imagem, pred in zip(imagens, previsoes):
                pred_item = pred.item()
                if len(imagens_coletadas[pred_item]) < 4:
                    imagens_coletadas[pred_item].append(imagem.cpu())
                    rotulos_previstos[pred_item].append(pred_item)

            if all(len(v) >= 4 for v in imagens_coletadas.values()):
                break

    fig, axs = plt.subplots(10, 4, figsize=(8, 12))
    for i in range(10):
        for j in range(4):
            axs[i, j].imshow(imagens_coletadas[i][j].squeeze(), cmap='gray')
            axs[i, j].set_title(f"{rotulos_previstos[i][j]}", fontsize=8)
            axs[i, j].axis('off')
        axs[i, 0].set_ylabel(f'{i}', rotation=0, labelpad=15, fontsize=12)

    plt.tight_layout()
    plt.savefig("util/classificacoes.png")
    plt.close()
    print("Gráfico de classificações salvo como classificacoes.png")

def plotar_precisoes(precisoes):
    rotulos = [str(i) for i in range(10)]
    plt.figure(figsize=(8, 4))
    plt.barh(rotulos, precisoes, color='skyblue')
    plt.xlim(0.90, 1.00)
    plt.xlabel("Precisão")
    plt.title("Precisão por Dígito (Zoom entre 90% e 100%)")
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for i, v in enumerate(precisoes):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig("util/precisao.png")
    plt.close()
    print("Gráfico de precisão salvo como precisao.png")