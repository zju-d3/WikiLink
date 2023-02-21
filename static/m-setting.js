


/////////////////////////////////////////-----  HTML DESIGN ONLY below -----///////////////////////////////////////////



function Hide_FuncPanel(){
    // document.getElementById("func-nav").style.display = "none";
    document.getElementById("point_show_results").style.display = "none";
    document.getElementById("line_show_results").style.display = "none";
};

function Show_FuncPanel(searchId){
    var search_To_funcP = {'keywords':'func-nav','point_textinput':'point_show_results','pathstart_textinput':'line_show_results','pathend_textinput':'line_show_results'};
    var panelId = search_To_funcP[searchId];

    //reset minhop and local or global
    reset_hops_switcher(panelId);

    d3.select('#'+panelId).style('display','block');
};

// CSS - search-box control
  function checkTextField(field) {
    if (field.value != '') {
        //d3.select('input[name="clear"]').style("visibility","visible");
        d3.select('input[name="search"]').style("opacity","0.8");
    }
    else{
        //d3.select('input[name="clear"]').style("visibility","hidden");
        d3.select('input[name="search"]').style("opacity","0.1");
        document.getElementById("func-nav").style.display = "none";
        Hide_FuncPanel();
        Hide_InfoPanel();
    }
  };

var clear_button=d3.select('input[name="clear"]');
    clear_button.on('click', function(){
        document.getElementById("func-nav").style.display = "none";
        document.getElementById("keywords").value = "";
        this.style.visibility = "hidden";
    });

// main search box setting //
 d3.selectAll("input#keywords").on('keydown',function(){
  var searchid=this.id;
  if (d3.event.keyCode==13){
      Handle_Search_Button(searchid);
  }else{
      document.getElementById("func-nav").style.display = "none";
  };
});

// exploreBox
 d3.selectAll("input#point_textinput").on('keydown',function(){
  var searchid=this.id;
  if (d3.event.keyCode==13){
      Handle_Search_Button(searchid);
  }else{
      Hide_FuncPanel();
      Hide_InfoPanel();
  };
});

 d3.selectAll("input#pathstart_textinput, input#pathend_textinput").on('keydown',function(){
  var seachid = this.id;
  if (d3.event.keyCode==13){
      Handle_pathSearchbutton(seachid);
  }else{
      Hide_FuncPanel();
      Hide_InfoPanel()
  };
});

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CSS - Global-local switch control
d3.selectAll('#switch-1,#switch-2,#switch-3').on('click',function(d){
    Hide_InfoPanel();
    if(this.id=='switch-3'){
        console.log('switch-3');
        resumeClusterColor();
    };
    if(this.id=='switch-1'){
        resetMinihop_Explore();
    };
    d3.select(this.parentNode.parentNode.parentNode).select('.t-global').classed('t-global-toggle',function(d){return !d3.select(this).classed('t-global-toggle');});
    d3.select(this.parentNode.parentNode.parentNode).select('.t-local').classed('t-local-toggle',function(d){return !d3.select(this).classed('t-local-toggle');});
});

/*$(document).ready(function(){
    $('#switch-1').click(function(){
        console.log('click$')
        $('.t-global').toggleClass('t-global-toggle');
        $('.t-local').toggleClass('t-local-toggle');
    });
    $('#switch-2').click(function(){
        console.log('click$')
        $('.t-global').toggleClass('t-global-toggle');
        $('.t-local').toggleClass('t-local-toggle');
    });
    $('#switch-3').click(function(){
        $('.t-global').toggleClass('t-global-toggle');
        $('.t-local').toggleClass('t-local-toggle');
    });
});*/

// selet explore minhops
function change_exploreMinhop(){
    Hide_InfoPanel();
};

// show func-nav panel
function show_panel(panelname){

    //update search box according to highlighted nodes
    var wid = d3.select('input#keywords').data()[0];
    var label = NODE_IdToObj(wid).label;
    document.getElementById("mainSearchBox").style.display = "none";
    document.getElementById("func-nav").style.display = "none";
    document.getElementById(panelname).style.display = "block";

    if (panelname == "point"){
        document.getElementById('point_show_results').style.display = "block";
        //reset minhop and local or global
        reset_hops_switcher('point_show_results');
        // update search box
        d3.select('input#point_textinput').node().value = label;
        // attach data
        d3.select('input#point_textinput').data([wid]);
    };


    if ( panelname == "line" ){
        // update search box
        d3.select('input#pathstart_textinput').node().value = label;
        d3.select('input#pathend_textinput').node().value='';
        // attach data
        d3.select('input#pathstart_textinput').data([wid]);
        d3.select('input#pathend_textinput').data([null]);
    };

    if(panelname == "cluster"){
        //cancel query highlight
        cancelQyHighlight();
    };

}

// show results panel
function show_info(panelname){
    if (panelname == "point"){
        console.log('point show');
        //document.getElementById("info_panel").style.height = "220px";
        Explore_showResult();
    }
    else if (panelname == "line"){
        console.log('line show');
        //document.getElementById("info_panel").style.top = "260px";
        FindPath_showResult();
    }
    else if (panelname == "cluster"){
        //document.getElementById("info_panel").style.top ="347px";
        BpathsClusters_showResult();
    };

    if(count_hideInfo % 2 == 1){
        console.log('count_hideInfo is even')
        count_hideInfo += 1;
        $('.info_panel_height').toggleClass('info_panel_down');
        $('.img-rotate').toggleClass('img-rotate-toggle');
    };
    document.getElementById("info_panel").style.display = "block";

}

// close line function panel
function closePanel(panelname){
    Hide_InfoPanel();
    Hide_FuncPanel();
    //cancelQyHighlight();
    var Qy = d3.select('input#keywords').data()[0];
    FOCUSING_NODE = Qy;
    var highlights = {'nodes':[Qy],'paths':[],'paths1':[]};
    highlight_nodespaths(highlights);

    if(panelname=='cluster') {
        cancelClusterColor();
    };
    document.getElementById("mainSearchBox").style.display = "block";
    document.getElementById("func-nav").style.display = "block";
    document.getElementById(panelname).style.display = "none";
}

// show the selected cluster setting option
function clusterSettingOption(){
    var option = document.getElementById("clusterMethod").value;
    if(option == "normalized"){
        document.getElementById("clusterMethod1Setting").style.display = 'block';
        document.getElementById("clusterMethod2Setting").style.display = 'none';
    }
    else if(option == "mcl"){
    document.getElementById("clusterMethod1Setting").style.display = 'none';
    document.getElementById("clusterMethod2Setting").style.display = 'block';
    };
}
// go back to cluster level 1 page
function backClusterLevel1(){
    document.getElementById("cluster_level_1").style.display = "block"
    document.getElementById("cluster_level_2").style.display = "none";
    //reset setting for Bpath panel
    d3.selectAll('#clusterStartSelect,#clusterEndSelect')
          .style('background-color','white')
          .each(function(d,i){
              this.value = "";
          });
    reset_hops_switcher('cluster');
    Hide_InfoPanel();
    resumeClusterColor();
};
// go to cluster level 2 page
function showClusterLevel2(){
    generate_Clusters();

    document.getElementById("cluster_level_1").style.display = "none";
    document.getElementById("cluster_level_2").style.display = "block";
    clusterPan("findPath");

}

// cluster input box control
$(document).ready(function(){
    var count = 1;
    $('#remBtn').hide();
    $("#addBtn").click(function(){
        var remove = "#field" + count;
        var iconCSS = 15+(count * 45);
        count = count + 1;
        var newIn = '<input id="field'+count+'" type="text" class="w3-input w3-border w3-round w3-hover-border-blue" style="height:45px;" placeholder="Point '+count+ '...">';
        var newS = '<div id="icon'+count+'" class="w3-display-topright material-icons w3-xxlarge cluster-searchicon" style="margin-top:'+iconCSS+'px; ">search</div>'
        var newInput = $(newIn);
        var newSBtn = $(newS);
        $("#field1").before(newInput);
        $("#field1").before(newSBtn);
        if (count == 5){
            $('#addBtn').hide();
        };
        if (count == 2){
            $('#remBtn').show();
        };
    });

    $('#remBtn').click(function(){
        var fieldID = "#field" + count;
        var iconID = "#icon" + count;
        $(iconID).remove();
        $(fieldID).remove();
        count = count - 1;
        if (count == 4){
            $('#addBtn').show();
        };
        if (count == 1){
            $('#remBtn').hide();
        };
    });

    /////////////////////// Fake drop-down list using jquery ///////////
    //////start path//////
    $('#clusterStartSelect').click(function(){
        $('#clusterStartList').slideToggle(150);
    });


    $('#clusterStartList').click(function(){
        $(this).slideUp(150);
    });

    $('#BpathStart').mouseleave(function(){
        $('#clusterStartList').slideUp(150);
    });

    //////End path////////
    $('#clusterEndSelect').click(function(){
        $('#clusterEndList').slideToggle(150);
    });


    $('#clusterEndList').click(function(){
        $(this).slideUp(150);
    });

    $('#BpathEnd').mouseleave(function(){
        $('#clusterEndList').slideUp(150);
    });
});

//show cluster func-panel

function clusterPan(clusterPanel){
    var i;
    var x = document.getElementsByClassName("cluster-panel");
    for (i=0;i < x.length; i++){
        x[i].style.display = "none";
    }
    document.getElementById(clusterPanel).style.display="block";
    Hide_InfoPanel();
    if(clusterPanel == "findPath"){
        d3.select("#clusterBtnPath").style("background-color","white");
        d3.select("#textPath").style("font-weight","bold");
        d3.select("#textPath").style("color","black");
        d3.select("#clusterBtnNode").style("background-color","#2196F3");
        d3.select("#textNode").style("font-weight","normal");
        d3.select("#textNode").style("color","white");
    }
    else if(clusterPanel =="generateMore"){
        d3.select("#clusterBtnNode").style("background-color","white");
        d3.select("#textNode").style("font-weight","bold");
        d3.select("#textNode").style("color","black");
        d3.select("#clusterBtnPath").style("background-color","#2196F3");
        d3.select("#textPath").style("font-weight","normal");
        d3.select("#textPath").style("color","white");
    };
};

//change the select of cluster
function ChangeSelectCluster(id){
    Hide_InfoPanel();
    resumeClusterColor();
    FOCUSING_CLUSTER = d3.select('#'+id).attr('value');
};


//////////////// Mobile design only ///////////////
var count_hideInfo = 0;
$(document).ready(function(){
    $('.hideInfo').click(function(){
        count_hideInfo += 1;
        console.log(count_hideInfo)
        $('.info_panel_height').toggleClass('info_panel_down');
        $('.img-rotate').toggleClass('img-rotate-toggle');
    });

    $('.fullInfo').click(function(){
        $('.info_panel_height').toggleClass('info_panel_full');
        $('.fullInfo').toggleClass('fullInfo_toggle');
    });
});