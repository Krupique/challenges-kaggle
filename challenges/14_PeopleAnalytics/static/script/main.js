$(document).ready(function() {

    $('#buttonSubmit').on('click', function() {

        $.ajax({
            //url: '/make_predict',
            url: "",
            type: 'post',
            //traditional: true,
            contenttype: "application/json; charset=utf-8",
            data: {
                nivel_satisfacao: $('#nivel_satisfacao').val(),
                tempo_empresa: $('#tempo_empresa').val(),
                numero_projetos: $('#numero_projetos').val(),
                horas_medias_por_mes: $('#horas_medias_por_mes').val(),
                ultima_avaliacao: $('#ultima_avaliacao').val()
            },
            success: function(result) {

                img_html = '<img src="/static/assets/null.png" id="imgPred">'

                if (result.result.previsao_value == "1") {
                    img_html = '<img src="/static/assets/fired.png" id="imgPred">'
                } else if (result.result.previsao_value == "0") {
                    img_html = '<img src="/static/assets/employed.png" id="imgPred">'
                }
                $('#divImg').html(img_html);


                //str_html = '<p>A previsão foi: ' + result.result.previsao + '</p>';
                //$('#resultado').append(str_html);

                console.log(result);
            },
            error: function(XMLHttpRequest, textStatus, errorThrown) {
                console.log("Error: " + errorThrown);
            },
            complete: function() {
                //alert('Chegou no complete!')
            }
        });
    });

});