from flask import Flask, request, render_template, jsonify
from tools import colaborador
import json

# Cria a app
app = Flask(__name__)

# Página index
@app.route('/')
def index():
    result = None
    return render_template('index.html', result=result)

@app.route("/make_predict",methods=["POST","GET"])
def make_predict():
    content_type = request.headers.get('Content-Type')

    teste = request.get_json()
    print(teste)

    print(content_type)
    if (content_type == 'application/json'):
        json = request.get_json()
        return json
    else:
        return 'Content-Type not supported!'


if __name__ == '__main__':
    app.run(debug=True)