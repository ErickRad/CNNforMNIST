import torch

def avaliar(modelo, dataloader, funcaoDePerda, device):
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    acertos_por_classe = [0] * 10
    total_por_classe = [0] * 10

    with torch.no_grad():
        for entradas, rotulos in dataloader:
            entradas, rotulos = entradas.to(device), rotulos.to(device)
            saidas = modelo(entradas)
            perda = funcaoDePerda(saidas, rotulos)
            perda_total += perda.item()
            _, previsto = torch.max(saidas, 1)
            acertos += (previsto == rotulos).sum().item()
            total += rotulos.size(0)

            for i in range(len(rotulos)):
                rotulo = rotulos[i].item()
                total_por_classe[rotulo] += 1
                if previsto[i].item() == rotulo:
                    acertos_por_classe[rotulo] += 1

    precisao_por_classe = [
        (acertos / total) if total > 0 else 0
        for acertos, total in zip(acertos_por_classe, total_por_classe)
    ]

    return perda_total / len(dataloader), acertos / total, precisao_por_classe