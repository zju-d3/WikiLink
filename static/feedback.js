$(document).ready(function(){

    //////////// Check Null question title from FB3 ////////////
    if($.trim($("#explore0").text())==''){
        console.log('explore0 is null')
        $('input[name="FE1-11"]').attr('value','null');
        $('input[name="FE1-12"]').attr('value','null');
        $("#explore0").parents('div.rating-layout').hide();
        $('#fb3-q1').hide();
        console.log('question1 hide')
    };

    if($.trim($("#explore1").text())==''){
        console.log('explore1 is null')
        $('input[name="FE1-21"]').attr('value','null');
        $('input[name="FE1-22"]').attr('value','null');
        $("#explore1").parents('div.rating-layout').hide();
        console.log('explore[1] hide')
    };

    if($.trim($("#explore2").text())==''){
        console.log('explore2 is null')
        $('input[name="FE1-31"]').attr('value','null');
        $('input[name="FE1-32"]').attr('value','null');
        $("#explore2").parents('div.rating-layout').hide();
        console.log('explore[2] hide')
    };

    if($.trim($("#path0").text())==''){
        console.log('path0 is null')
        $('input[name="FE2-1"]').attr('value','null');
        $('#fb3-q2').hide();
        console.log('question2 hide')
    };

    if($.trim($("#path1").text())==''){
        console.log('path1 is null')
        $('input[name="FE2-2"]').attr('value','null');
        $("#path1").parents('div.rating-layout').hide();
        console.log('path[1] hide')
    };

    if($.trim($("#cluster").text())==''){
        console.log('cluster is null')
        $('input[name="FE4"]').attr('value','null');
        $('#fb3-q4').hide();
        console.log('question4 hide')
    };

    ///////////  Rating system design  ////////////
    $('circle[stroke="#8600b3"]').click(function(){
        var inputName = $(this).attr("class");
        var value = $(this).attr('value');
        $('input[name='+ inputName +']').attr('value',value);
        $(this).siblings().attr('fill','white');
        $(this).attr('fill','#8600b3');
    });
    $('circle[stroke="#cccccc"]').click(function(){
        var inputName = $(this).attr("class");
        var value = $(this).attr('value');
        $('input[name='+ inputName +']').attr('value',value);
        $(this).siblings().attr('fill','white');
        $(this).attr('fill','#cccccc');
    });
    $('circle[stroke="#4CAF50"]').click(function(){
        var inputName = $(this).attr("class");
        var value = $(this).attr('value');
        $('input[name='+ inputName +']').attr('value',value);
        $(this).siblings().attr('fill','white');
        $(this).attr('fill','#4CAF50');
    });

    ////////////  OPTION 'OTHER' DESIGN /////////////////
    $('#BI5-OTHER').click(function(){
        if($(this).is(':checked')){
            $('#BI5-INPUT').css('display','block');
        }
        else{
            $('#BI5-INPUT').css('display','none');
        }
    });


    $('input[name$="HCI3"]').click(function(){
        if($('#HCI3_other').is(':checked')){
            $('#HCI3_input').css('display','block');
        }
        else{
            $('#HCI3_input').css('display','none');
        }
    });

    $('input[name$="FE5"]').click(function(){
        if($('#FE5_yes').is(':checked')){
            $('#FE5_rate').css('display','block');
        }
        else{
            $('#FE5_rate').css('display','none');
        }
    });

    $('input[name$="FE6"]').click(function(){
        if($('#FE6_yes').is(':checked')){
            $('#FE6_rate').css('display','block');
        }
        else{
            $('#FE6_rate').css('display','none');
        }
    });

    $('input[name$="FE7"]').click(function(){
        if($('#FE7_no').is(':checked')){
            $('#FE7reason').show();
        }
        else{
            $('#FE7reason').hide();
        }
    });

    ////////////  Required form submission CHECK /////////////////
    $('form').submit(function(){

        var required = $('[required]');
        var error = false;

        for(var i = 0; i <= (required.length - 1);i++)
        {
            if(required[i].value == '') // tests that each required value does not equal blank, you could put in more stringent checks here if you wish.
            {
                required[i].style.backgroundColor = 'rgb(255,155,155)';
                error = true; // if any inputs fail validation then the error variable will be set to true;
            }
            else if(required[i].type == 'radio')
            {   group = required[i].name
                if ($('input[name='+group+']').is(':checked')){
                }
                else{
                    error = true;
                    required[i].style.backgroundColor = 'rgb(255,155,155)';
                }
            }
        };

        if(error) // if error is true;
        {
            alert('Please answer the required questions, thanks !')
            return false;  // stop the form from being submitted.
        };
    });

});


/*function otherJob(){

    var fb_job = document.getElementById("job");
    var fb_job_option = fb_job.options[fb_job.selectedIndex].value;

    if (fb_job_option == "other"){
        document.getElementById("job_other").style.display = "block";
    }
    else
        document.getElementById("job_other").style.display = "none";
}

function otherSubj(){

    var fb_subj = document.getElementById("subject");
    var fb_subj_option = fb_subj.options[fb_subj.selectedIndex].value;

    if (fb_subj_option == "other"){
        document.getElementById("subject_other").style.display = "block";
    }
    else
        document.getElementById("subject_other").style.display = "none";
}
*/


