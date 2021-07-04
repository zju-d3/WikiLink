//update graph for search button
//query is the node to be searched or added
// zh_nodes is the list of nodes to be zoomed and highlighted as end nodes
function updateGraph_searchButton(query){
    var currentnodes = CURRENT_NODESSET(CLIENT_NODES,"wid");
    if( _.difference(query,currentnodes).length==0 ){
        var highlights = {'nodes':query,'paths':[],'paths1':[]};
        highlight_nodespaths(highlights);
        ZoomToNodes(query);
    }else{
        var info={'currentnodes':currentnodes,'query':query, 'tp':Type_distance};
        d3.json('/searchbutton/'+JSON.stringify(info),function(error,data){
            if(data.bornnode){
                var bornnode = CLIENT_NODES.filter(function(obj){return obj["wid"]==data.bornnode;})[0];
                var bornplace = {x:bornnode.x, y:bornnode.y, vx:bornnode.vx, vy: bornnode.vy};
            }else{var bornplace = {x:w/2, y:h/2, vx:NaN, vy: NaN};};
            SHOW_UPDATE_FORCE(data,bornplace);
            node_left_click_on();
            var highlights = {'nodes':query,'paths':[],'paths1':[]};
            highlight_nodespaths(highlights);
            ZoomToNodes(query);
        });
    };

    //update previous focus
    FOCUSING_NODE = query[0];

};


// Handle search button
function Handle_Search_Button(searchbutton_id){
    var info = {'searchtext':get_inputtext(searchbutton_id)};
    d3.json('/texttowid/'+JSON.stringify(info),function(error,data){
//        console.log("data:")
//        console.log(data)
//        console.log(data.toString().length)
        if (data || data==0){
            d3.select('input#' + searchbutton_id ).data([data]);
            updateGraph_searchButton([data]);

            Show_FuncPanel(searchbutton_id);
        }else{
            alert('Can not match your input concepts');
        };
    });

};

// handle search button of path
function Handle_pathSearchbutton(searchbutton_id){
     var info = {'searchtext':get_inputtext(searchbutton_id)};
     d3.json('/texttowid/'+JSON.stringify(info),function(error,data){
         if (data.toString().length>0){
             var node1=data;
             var selector1='input#'+searchbutton_id;
             d3.select(selector1).data([node1]);
             var theother_id= _.difference( ['pathstart_textinput','pathend_textinput'] , [searchbutton_id])[0];

             var info1 = {'searchtext':get_inputtext(theother_id)};
             d3.json('/texttowid/'+JSON.stringify(info1),function(error,data){
                 if(data || data==0){
                     var node2=data;
                     var selector2='input#'+theother_id;
                     d3.select(selector2).data([node2]);
                     var queries=[node1,node2];
                     if( node1!=node2 ){
                         Show_FuncPanel(searchbutton_id);
                     };
                 }else{
                     var queries=[node1];
                 };
                 updateGraph_searchButton(queries);
             });
         }else{
             alert('Can not match this input');
         };
     });
};

//Explore show results
function Explore_showResult(){
        var query = d3.select('input#point_textinput').data()[0];
        var label = NODE_IdToObj(query).label;
        var minhops = get_minhops('minhop_point');
        Explore_Nearby(check_explore_LG('switch-1'),true,minhops,N_SearchButton,query,query);
        //update input search box
        d3.select('input#point_textinput').node().value = label;
        //onclick next and previous
        d3.select('#info_panel #pageup').on('click', Explore_Previous);
        d3.select('#info_panel #pagedown').on('click', Explore_Next);
};

//Explore next handler OK
function Explore_Next(){
    var query = d3.select('input#point_textinput').data()[0];
    var minhops = get_minhops('minhop_point');
    Explore_Nearby(check_explore_LG('switch-1'),false,minhops,N_SearchButton,query,query);
};

//Explore previous handler OK
function Explore_Previous(){
    var query = d3.select('input#point_textinput').data()[0];
    var minhops = get_minhops('minhop_point');
    Explore_Nearby(check_explore_LG('switch-1'), false, minhops, -N_SearchButton, query, query);
};

// Findpath show results
function FindPath_showResult(){
      var node1 = d3.select('input#pathstart_textinput').data()[0];
      var node2 = d3.select('input#pathend_textinput').data()[0];
      var label1 = NODE_IdToObj(node1).label;
      var label2 = NODE_IdToObj(node2).label;

      var minhops = get_minhops('minhop_line');
      findPaths_betweenNodes(check_explore_LG('switch-2'), true, minhops, N_forPath, node1, node2);
      // update search box of start and end point
      d3.select('input#pathstart_textinput').node().value = label1;
      d3.select('input#pathend_textinput').node().value = label2;
      // onclick next and previous
      d3.select('#info_panel #pageup').on('click', FindPath_previous);
      d3.select('#info_panel #pagedown').on('click', FindPath_next);
};

//find path next
function FindPath_next(){
    var node1 = d3.select('input#pathstart_textinput').data()[0];
    var node2 = d3.select('input#pathend_textinput').data()[0];
    var minhops = get_minhops('minhop_line');
    findPaths_betweenNodes(check_explore_LG('switch-2'), false, minhops, N_forPath, node1, node2);
};
//find path previous
function FindPath_previous(){
    var node1 = d3.select('input#pathstart_textinput').data()[0];
    var node2 = d3.select('input#pathend_textinput').data()[0];
    var minhops = get_minhops('minhop_line');
    findPaths_betweenNodes(check_explore_LG('switch-2'), false, minhops, -N_forPath, node1, node2);
};

//B-paths show results
function BpathsClusters_showResult(){
    var value1 = d3.select('#clusterStartSelect').node().value;
    var cluster1 = d3.select('#clusterStartList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value')==value1;})
                     .data()[0];

    var value2 = d3.select('#clusterEndSelect').node().value;
    var cluster2 = d3.select('#clusterEndList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value') == value2;})
                     .data()[0];

    findBpaths_betweenClusters(check_explore_LG('switch-3'), true, N_forPath, cluster1, cluster2);

    //onclick next and previous
    d3.select('#info_panel #pageup').on('click', Bpaths_Previous);
    d3.select('#info_panel #pagedown').on('click', Bpaths_Next);
};
//B-paths next
function Bpaths_Next(){
    var value1 = d3.select('#clusterStartSelect').node().value;
    var cluster1 = d3.select('#clusterStartList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value')==value1;})
                     .data()[0];

    var value2 = d3.select('#clusterEndSelect').node().value;
    var cluster2 = d3.select('#clusterEndList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value') == value2;})
                     .data()[0];

    findBpaths_betweenClusters(check_explore_LG('switch-3'), false, N_forPath, cluster1, cluster2);
};
//B-paths previous
function Bpaths_Previous(){
    var value1 = d3.select('#clusterStartSelect').node().value;
    var cluster1 = d3.select('#clusterStartList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value')==value1;})
                     .data()[0];

    var value2 = d3.select('#clusterEndSelect').node().value;
    var cluster2 = d3.select('#clusterEndList')
                     .selectAll('a')
                     .filter(function(d){return d3.select(this).attr('value') == value2;})
                     .data()[0];

    findBpaths_betweenClusters(check_explore_LG('switch-3'), false, -N_forPath, cluster1, cluster2);
};

// node click behavior
function node_right_click_on(){
      GRAPH.selectAll('.gnode').on('contextmenu',function(d){
      d3.event.preventDefault();
      console.log("click node");
      console.log(d.label);
      d3.json('/neighbor_level/'+d.wid,function(error,data){
          console.log(JSON.stringify(data));
          circle_layout_neighbor(data);
          Backlayer_clickon();
      });
   });
};

// node left click behavior
function node_left_click_on(){
      GRAPH.selectAll('.gnode').on('mouseover',function(){
          d3.selectAll('input#keywords,input#point_textinput,input#pathstart_textinput,input#pathend_textinput').each(function(d){this.blur();});
      });

      GRAPH.selectAll('.gnode').on('click',function(d){
          Hide_InfoPanel();
          var clicked_data = d;
          var preNode =  FOCUSING_NODE;
          var preLabel = NODE_IdToObj(preNode).label;
          FOCUSING_NODE = d.wid;

          if( d3.select('#line').style('display')=='block'){
              if(preNode == clicked_data.wid || d3.select('input#pathstart_textinput').data()[0] == d3.select('input#pathend_textinput').data()[0]){
                  //highlight
                  var highlights = {'nodes':[clicked_data.wid],'paths':[],'paths1':[]};
                  highlight_nodespaths(highlights);
                  //update searchbox
                  d3.select('input#pathstart_textinput').node().value = clicked_data.label;
                  d3.select('input#pathend_textinput').node().value='';
                  //attach data
                  d3.select('input#pathstart_textinput').data([clicked_data.wid]);
                  d3.select('input#pathend_textinput').data([null]);
              }else{
                  //highlight
                  var highlights = {'nodes':[clicked_data.wid,preNode],'paths':[],'paths1':[]};
                  highlight_nodespaths(highlights);

                  d3.selectAll('input#pathstart_textinput,input#pathend_textinput')
                    .each(function(d){
                        if(d==preNode){
                            //update search box
                            this.value = preLabel;
                        }else{
                            //update search box
                            this.value = clicked_data.label;
                            //data attach
                            d3.select(this).data([clicked_data.wid]);
                        };
                    });

                  Show_FuncPanel('pathstart_textinput');
              };

          }else if( d3.select('#point').style('display')=='block' || d3.select('#mainSearchBox').style('display')=='block' ){
              // highlight the clicked node
              var highlights = {'nodes':[d.wid],'paths':[],'paths1':[]};
              highlight_nodespaths(highlights);
              // reset explore minihop
              resetMinihop_Explore();

              if(d3.select('#point').style('display')=='block'){ var searchid='point_textinput';}
              if(d3.select('#mainSearchBox').style('display')=='block'){ var searchid='keywords';};
              //update inputbox
              d3.select('input#'+searchid).node().value = d.label;
              //attach data
              d3.select('input#'+searchid).data([d.wid]);
          } else if( d3.select('#cluster_level_2').style('display')=='block' ){
              cancelInfoHighlight();
              resumeClusterColor();
              var color = d3.select(this).select('circle').datum().color;

              if(color){
                  var preCluster=FOCUSING_CLUSTER;
                  FOCUSING_CLUSTER = clicked_data.icluster;

                  if( preCluster==FOCUSING_CLUSTER ||  d3.select('#clusterStartSelect').node().value==d3.select('#clusterEndSelect').node().value){
                      //color
                      d3.select('#clusterStartSelect').style('background-color', color);
                      d3.select('#clusterEndSelect').style('background-color','white');
                      //value
                      d3.select('#clusterStartSelect').node().value = FOCUSING_CLUSTER;
                      d3.select('#clusterEndSelect').node().value = "";

                  }else{
                      d3.selectAll('#clusterStartSelect,#clusterEndSelect')
                        .filter(function(d){return parseInt(this.value)!=preCluster;})
                        .each(function(d,i){
                                assert(i!=1, 'precluster not in cluster selection box');
                                d3.select(this).style('background-color',color); //color
                                this.value = FOCUSING_CLUSTER; //value

                        });
                  };
              }
          };
   });
};

// Backlayer background click on
function Backlayer_clickon(){
    BACKLAYER.on('click',function(){
        console.log("click Backlayer");
        RedoBack();
    });
};



