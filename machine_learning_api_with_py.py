import warnings

from flask import Flask
from textblob import TextBlob

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
    return f"polaridade: {polaridade}"


app.run(debug=True)

#restart a execução automaticamente.
#app.run(debug=True)