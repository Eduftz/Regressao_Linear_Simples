# Importações necessárias para o projeto
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from pathlib import Path


# Classe criada para executar as funções
class RegressaoSimples:
    def __init__(self, x, y, idade):
        self.x = x
        self.y = y
        self.idade = idade

        # Tratamento para caso seja inserido um numero já utilizado
        if not np.isin(self.idade, self.x):
            self.idade = idade
        else:
            raise 'Idade já está em uso favor escolher outra'

    def correlacao(self):
        raiz_quadrada = np.sqrt(np.var(self.x) * np.var(self.y))
        covarianca = np.cov(self.x, self.y, bias=True)[0][1]
        return covarianca / raiz_quadrada

    def inclinacao(self):
        desvio_x = np.std(self.y)
        desvio_y = np.std(self.x)
        return self.correlacao() * (desvio_x / desvio_y)

    def interceptacao(self):
        media_y = np.mean(self.y)
        media_x = np.mean(self.x)
        return media_y - self.inclinacao() * media_x

    def previsao(self):

        # Foi utilizado o round para arredondar o resultado
        return round(self.interceptacao() + self.inclinacao() * self.idade)

    # novo dataset criado para receber os dados da previsão, gerando um documento no formato csv localmente
    def novo_dataset(self):
        add_idade = np.sort(np.append(self.idade, self.x))
        add_valor = np.sort(np.append(self.previsao(), self.y))
        dados = pd.DataFrame({'Idade': add_idade,
                              'Valor': add_valor})
        filepath = Path('documento_atualizado.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return dados.to_csv(filepath, index=False)

    # Função para o plot do gráfico
    def grafico(self):
        novo_dataset = pd.read_csv('documento_atualizado.csv')
        plt.figure(figsize=(8, 6))
        sb.regplot(x='Idade',
                   y='Valor',
                   ci=None,
                   data=novo_dataset,
                   color='red',
                   scatter_kws={'color': 'blue'},
                   line_kws={'color': 'black'})
        plt.title('Regressão Linear Simples', fontsize=10)
        plt.xlabel('Idade', fontsize=10)
        plt.ylabel('Valor', fontsize=10)
        plt.legend(['Valores', 'Linha de melhor ajuste'],
                   loc='lower right', bbox_to_anchor=(1.13, -0.14), fontsize=8)
        return plt.show()


# Para executar aqui mesmo e não ser usado como módulo.
if __name__ == '__main__':

    # Digite o caminho em que se encontra o arquivo csv, lembre-se de colocar o caminho e o nome do arquivo também.
    dataset = pd.read_csv(input('Coloque o caminho: '))

    '''
    O nome dentro do dataset refere-se as colunas, no caso a primeira coluna seria Idade e a segunda Valor do documento
    Convenio.csv. Se quiser testar previsões com outras variaveis, deve-se trocar o nome das colunas dentro do Convenio.csv 
    OU pode-se usar outro arquivo trocando o nomedos respectivos dataset abaixo, pois se não for feito o ocorrerá erro.
    '''
    posicao_x = dataset['Idade']
    x_array = np.array(posicao_x)
    posicao_y = dataset['Valor']
    y_array = np.array(posicao_y)

    idade_previsao = input('Coloque a idade que deseja prever o valor: ')

    # Tratamento básico caso seja digitado letra ao invés de numero
    if idade_previsao.isnumeric():
        idade_previsao = int(idade_previsao)
    else:
        raise 'Coloque somente numeros!'

    regressao = RegressaoSimples(x_array, y_array, idade_previsao)
    regressao.novo_dataset()

    # impressao direto no console
    print(f'Com a idade de {idade_previsao} anos, a previsão de fica em R${regressao.previsao()}')

    regressao.grafico()
