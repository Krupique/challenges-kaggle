$(document).ready(function() {

    $('#form').on('submit', function() {

        text1 = $('#input1').val()
        text2 = $('#input2').val()

        $.ajax({
            url: '/make_predict',
            type: 'post',
            data: JSON.stringify({
                campo1: text1,
                campo2: text2
            }),
            contentType: false,
            processData: false,
            success: function(result) {
                //console.log(result)
                alert('Chegou aqui')
                console.log(result)
            },
        });
    });

});