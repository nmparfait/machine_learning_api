import warnings
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle

# use the serialization with pickle in order to upload file with speed
modelo = pickle.load(open('modelo.sav', 'rb'))

colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'parfait'
app.config['BASIC_AUTH_PASSWORD'] = 'mlops'

basic_auth = BasicAuth(app)
@app.route('/')
def home():
    return "Minha primeira API em python"


# url para ponto de accesso
@app.route("/sentimento/<frase>")
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to="en")
    warnings.warn("this is deprecated", DeprecationWarning, 2)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route("/cotacao/", methods=['POST'])
@basic_auth.required
def cotacao(tamanho):
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)

#restart a execução automaticamente.
#app.run(debug=True)