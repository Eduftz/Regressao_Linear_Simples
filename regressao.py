import numpy as _np
import matplotlib.pyplot as _plt
import seaborn as _sb
import pandas as _pd
from pathlib import Path


class RegressaoSimples:
    def __init__(self, x, y, idade):
        self.x = x
        self.y = y
        self.idade = idade
        if not _np.isin(self.idade, self.x):
            self.idade = idade
        else:
            raise 'Idade já está em uso favor escolher outra'

    def correlacao(self):
        raiz_quadrada = _np.sqrt(_np.var(self.x) * _np.var(self.y))
        covarianca = _np.cov(self.x, self.y, bias=True)[0][1]
        return covarianca / raiz_quadrada

    def inclinacao(self):
        desvio_x = _np.std(self.y)
        desvio_y = _np.std(self.x)
        return self.correlacao() * (desvio_x / desvio_y)

    def interceptacao(self):
        media_y = _np.mean(self.y)
        media_x = _np.mean(self.x)
        return media_y - self.inclinacao() * media_x

    def previsao(self):
        return round(self.interceptacao() + self.inclinacao() * self.idade)

    def novo_dataset(self):
        add_idade = _np.sort(_np.append(self.idade, self.x))
        add_valor = _np.sort(_np.append(self.previsao(), self.y))
        valors = _pd.DataFrame({'Idade': add_idade,
                                'Valor': add_valor})
        filepath = Path('documento_atualizado.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return valors.to_csv(filepath, index=False)

    def grafico(self):
        novo_dataset = _pd.read_csv('documento_atualizado.csv')
        _plt.figure(figsize=(8, 6))
        _sb.regplot(x='Idade',
                    y='Valor',
                    ci=None,
                    data=novo_dataset,
                    color='red',
                    scatter_kws={'color': 'blue'},
                    line_kws={'color': 'black'})
        _plt.title('Regressão Linear Simples', fontsize=10)
        _plt.xlabel('Idade', fontsize=10)
        _plt.ylabel('Valor', fontsize=10)
        _plt.legend(['Valores', 'Linha de melhor ajuste'],
                    loc='lower right', bbox_to_anchor=(1.13, -0.14), fontsize=8)
        return _plt.show()


if __name__ == '__main__':
    dataset = _pd.read_csv(input('Coloque o caminho: '))
    posicao_x = dataset['Idade']
    x_array = _np.array(posicao_x)
    posicao_y = dataset['Valor']
    y_array = _np.array(posicao_y)
    idade_previsao = input('Coloque a idade que deseja prever o valor: ')
    if idade_previsao.isnumeric():
        idade_previsao = int(idade_previsao)
    else:
        raise 'Coloque somente numeros!'
    regressao = RegressaoSimples(x_array, y_array, idade_previsao)
    regressao.novo_dataset()
    print(f'Com a idade de {idade_previsao} anos, a previsão de fica em R${regressao.previsao()}')
