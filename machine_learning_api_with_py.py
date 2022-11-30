import warnings
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

df = pd.read_csv('casas.csv')

# escolher apenas as colunas que usaremos para criar o nosso modelo
colunas = ['tamanho', 'ano', 'garagem']
#df = df[colunas]

# valor preditor
X = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

app = Flask(__name__)


@app.route('/')
def home():
    return "Minha primeira API em python"


# url para ponto de accesso
@app.route("/sentimento/<frase>")
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to="en")
    warnings.warn("this is deprecated", DeprecationWarning, 2)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route("/cotacao/", methods=['POST'])
def cotacao(tamanho):
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)

#restart a execução automaticamente.
#app.run(debug=True)