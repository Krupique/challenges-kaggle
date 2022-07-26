from flask import Flask, request, render_template, jsonify
from tools.colaborador import Colaborador
import json


# Cria a app
app = Flask(__name__)

# Página index
@app.route('/',  methods=["POST", "GET"])
def index():
    
    if request.method == 'POST':
        nivel_satisfacao = request.form['nivel_satisfacao']
        tempo_empresa = request.form['tempo_empresa']
        numero_projetos = request.form['numero_projetos']
        horas_medias_por_mes = request.form['horas_medias_por_mes']
        ultima_avaliacao = request.form['ultima_avaliacao']

        status, nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao = realizarVerificacoes(nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao) 

        if status == 'OK':
            colaborador = Colaborador(nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao)
            previsao = colaborador.predict()
        else:
            previsao = -1

        previsao = previsao[0]
        previsao_value = str(previsao)
        if previsao == 0:
            previsao = 'O colaborador não irá deixar a empresa!'
        elif previsao == 1:
            previsao = 'O colaborador irá deixar a empresa!'
        else:
            previsao = 'Valor inesperado'

        return jsonify(result={'previsao': previsao, 'previsao_value': previsao_value})


    return render_template('index.html')

def realizarVerificacoes(nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao):
    nivel_satisfacao = float(nivel_satisfacao)
    tempo_empresa = float(tempo_empresa)
    numero_projetos = float(numero_projetos)
    horas_medias_por_mes = float(horas_medias_por_mes)
    ultima_avaliacao = float(ultima_avaliacao)

    return 'OK', nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao


if __name__ == '__main__':
    app.run(debug=True)