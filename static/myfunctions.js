//assert like python
function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}



// get the text in inputbox
function get_inputtext(searchbutton_id){
  var selector='input#'+searchbutton_id
  return d3.select(selector).node().value;
};

// Get an array which consists of the values of a paticular key of objects in another array
function CURRENT_NODESSET(nodes,key){
    var nodesset=[];
    nodes.forEach(function(d){
        nodesset.push(d[key]);
    });
    return nodesset;
};

// if the node exists in current CLIENT_NODES, transfer the id to object
function NODE_IdToObj(id){
    var cnode = CLIENT_NODES.filter(function(d){return d.wid==id;});
    assert(cnode.length==1, 'no such node id in current client nodes')
    return cnode[0];
};


// update force layout
function SHOW_UPDATE_FORCE(dataset,born){

  // born postion and velocity of new nodes
  /*born = bornplace
  if(query_exist_in_local){
      var born = CLIENT_NODES.filter(function(obj){return obj["wid"]==dataset.query;})[0];
  }else if(CLIENT_NODES.length==0){
      var born = {x:NaN,y:NaN,vx:NaN,vy:NaN};
  }else{
      var born = {x:w/2, y:h/2, vx:NaN, vy: NaN};
  };*/

  //update and add nodes
  dataset.allnodes.forEach(function(d){
      var cnode = CLIENT_NODES.filter(function(obj){return obj['wid']==d.wid;});

      if(cnode.length==0){//add new nodes
          d.x = born.x;
          d.y = born.y;
          d.vx = born.vx;
          d.vy = born.vy;
          CLIENT_NODES.push(d);
          CLIENT_NODES_ids.push(d.wid);
      }else{ // update existing nodes degree
          cnode=cnode[0];
          cnode.n=d.n;
          cnode.N=d.N;
      };
  });


  CLIENT_EDGES=dataset.alledges;
  // update simulation
  SIMULATION.nodes(CLIENT_NODES);
  SIMULATION.force("link").links(CLIENT_EDGES);


  //scale
  scale_tp2Distance = d3.scalePow().exponent(-2)
                   .domain([ d3.min(CLIENT_EDGES,function(d){return d['dist']})+1.0, d3.max(CLIENT_EDGES,function(d){return d['dist']})+1.0  ])
                   .range([minlinkdistance,maxlinkdistance]);

  scale_tp2Stokewidth = d3.scalePow().exponent(-2)
                    .domain([ d3.min(CLIENT_EDGES,function(d){return d['dist']})+1.0, d3.max(CLIENT_EDGES,function(d){return d['dist']})+1.0  ])
                    .range([maxlinkwidth,minlinkwidth]);

  scale_NodeRadius = d3.scalePow().exponent(3)
                        .domain([ d3.min( CLIENT_NODES , function(d){return d.N}) , d3.max( CLIENT_NODES , function(d){return d.N}) ])
                        .range([minNodeRadius,maxNodeRadius]);

  //update link distance
  SIMULATION.force("link").distance(function(d){return scale_tp2Distance(d['dist']+1.0);});

  //change title color
  TITLECOLOR_CHANGE();
  //update svg
  var edges=GRAPH.selectAll(".edge")
               .data(SIMULATION.force("link").links(),function(d){return Math.min(d.source.wid,d.target.wid)+"-"+Math.max(d.source.wid,d.target.wid);});

          edges.attr("stroke-width",function(d){return scale_tp2Stokewidth(d['dist']+1.0);});
          edges.enter()
               .insert("line",":first-child")
               .attr("class","edge")
               .attr("stroke-width",function(d){return scale_tp2Stokewidth(d['dist']+1.0);});
          edges.exit().remove();

  /*var edgelabels=GRAPH.selectAll(".edgelabel")
                    .data(SIMULATION.force("link").links(),function(d){return Math.min(d.source.wid,d.target.wid)+"-"+Math.max(d.source.wid,d.target.wid);})
                    .text(function(d){return Number((d['dist']).toFixed(1));});

          edgelabels.enter()
                    .append("text")
                    .attr("class","edgelabel")
                    .text(function(d){return Number((d['dist']).toFixed(1));});
          edgelabels.exit().remove();*/

  var gnodes = GRAPH.selectAll(".gnode")
               .data(SIMULATION.nodes(),function(d){return d.wid;});

      gnodes.select("circle").transition('Radius').attr("r",function(d){return scale_NodeRadius(d.N);});

  var newgnodes=gnodes.enter()
               .append("g")
               .attr("class","gnode")
               .call(d3.drag()
               .on("start", dragstarted)
               .on("drag", dragged)
               .on("end", dragended));

  newgnodes.append("circle").transition('Radius')
         .attr("r",function(d){return scale_NodeRadius(d.N);});
  newgnodes.append("text")
         .attr("dy",-10)
         .text(function(d){return d.label;});

  gnodes.exit().remove();

  // restart simulation
  SIMULATION .alphaTarget(0.5).restart();

};

// transform localgraph to circle-layout according to the neighbor-level
function circle_layout_neighbor(dataset){
   var level = dataset["disconnected"].length? Object.keys(dataset).length-1 : Object.keys(dataset).length-2 ;
   scale_Rnei=d3.scaleLinear().domain([1,level]).range([d3.min([w,h])/4,d3.min([w,h])/2-maxNodeRadius]);

   // remove force
   SIMULATION.force("link").strength(0);
   SIMULATION.force("center",null);
   SIMULATION.force("charge",null);

   function Get_CoordX(cx,i,n,R,l){
       var theta=2*Math.PI/level*(l-1)+2*Math.PI/n*i;
       var x=cx+ R*Math.cos(theta);
       return Math.round(x);
   };
   function Get_CoordY(cy,i,n,R,l){
       var theta=2*Math.PI/level*(l-1)+2*Math.PI/n*i;
       var y=cy- R*Math.sin(theta);
       return Math.round(y);
   };

   SIMULATION.force("xp",d3.forceX());
   SIMULATION.force("xp").x(function(d,i){
       for (var key in dataset){
           if (dataset[key].indexOf(d.wid)>=0){
               var l = key=="disconnected"? level:key ;
               var i = dataset[key].indexOf(d.wid);
               var n = dataset[key].length;
               break;
           };
       };

       if(l==0){
           var R=0;
       }else{
           var R=scale_Rnei(l)
       };

       var coor_x=Get_CoordX(w/2,i,n,R,l);
       return coor_x;
   })
   .strength(POSITIONFORCE_STRENGTH);

   SIMULATION.force("yp",d3.forceY());
   SIMULATION.force("yp").y(function(d,i){
       for (var key in dataset){
           if (dataset[key].indexOf(d.wid)>=0){
               var l = key=="disconnected"? level:key;
               var i = dataset[key].indexOf(d.wid);
               var n = dataset[key].length;
               break;
           };
       };

       if(l==0){
           var R=0;
       }else{
           var R=scale_Rnei(l)
       };

       var coor_y=Get_CoordY(h/2,i,n,R,l);
       return coor_y;
   })
   .strength(POSITIONFORCE_STRENGTH);

   d3.selectAll(".neighbor_track").remove();
   neighbor_tracks=GRAPH.selectAll(".neighbor_track")
                      .data(_.range(1,level+1))
                      .enter()
                      .append("circle")
                      .attr("class","neighbor_track")
                      .attr("cx",w/2)
                      .attr("cy",h/2)
                      .moveToBack()
                      .transition(tran)
                      .attr("r",function(d){return scale_Rnei(d);});


   SIMULATION.alphaTarget(0.3).restart();
   //change title color
   TITLECOLOR_CHANGE();
};

//dataset.nodes are the nodes the queries
//dataset.paths are the paths to be focused
//dataset.paths1 are the unfocused paths which are highligthed
 //hltQ = dataset.nodes
 //hltA = (start,end) of dataset.paths
 //hltP = middle part of dataset.paths
 //hltA1 = (start,end) of dataset.paths1
 //hltP1 = middle part of dataset.paths1
function highlight_nodespaths(dataset){
    //is it in cluster panel?
    var clusterpanel = (d3.select('#cluster_level_2').style('display')=='block');
    if (clusterpanel){
        var icluster1 = d3.select('#clusterStartSelect').node().value;
        var icluster2 = d3.select('#clusterEndSelect').node().value;
        var targetClusters = [parseInt(icluster1),parseInt(icluster2)];
    };

    // generate a highlighted graph based on dataset.paths
    var hltQ= dataset.nodes.slice();
    var hltA=[];
    var hltG = new jsnx.Graph();
    for (var i = 0; i < dataset.paths.length; i++){
        hltG.addPath(dataset.paths[i]);
        hltA = _.union( hltA , [dataset.paths[i][0], dataset.paths[i].slice(-1)[0] ] );
    };
    var hltA1 = [];
    var hltG1=new jsnx.Graph();
    for (var i = 0; i < dataset.paths1.length; i++){
        hltG1.addPath(dataset.paths1[i]);
        hltA1.push(dataset.paths1[i][0]);
        hltA1.push(dataset.paths1[i].slice(-1)[0]);
    };


    if(clusterpanel){
        // all nodes color
        d3.selectAll(".gnode").selectAll("circle").each(function(d){
            if( _.contains(hltQ, d.wid) ){
                d3.select(this).attr('class','hltQ');
            }else if( _.contains(hltA, d.wid) ){
                d3.select(this).attr('class','hltA');
            }else if( hltG.hasNode(d.wid) ){
                d3.select(this).attr('class','hltP').style('fill',null);
            }else if( _.contains(hltA1, d.wid) ){
                d3.select(this).attr('class','hltA1');
            }else if( hltG1.hasNode(d.wid) ){
                d3.select(this).attr('class','hltP1').style('fill',null);
            }else{
                if(_.contains(targetClusters,d.icluster)){
                    d3.select(this).attr('class','');
                }else{
                    d3.select(this).attr('class','').style('fill',null);
                };
            };
        });
        //all edges color
        d3.selectAll(".edge").each(function(d){
            if( hltG.hasEdge(d.source.wid,d.target.wid) ){
                d3.select(this).attr('class','edge hltE').style('stroke',null);
            }else if( hltG1.hasEdge(d.source.wid,d.target.wid) ){
                d3.select(this).attr('class','edge hltE1').style('stroke',null);
            }else{
                if( !(d.source.icluster == d.target.icluster && _.contains(targetClusters,d.source.icluster)) ){
                    d3.select(this).attr('class','edge').style('stroke',null);
                };
            };
        });

    }else{
        // all nodes color
        d3.selectAll(".gnode").selectAll("circle").each(function(d){
            if( _.contains(hltQ, d.wid) ){
                d3.select(this).attr('class','hltQ');
            }else if( _.contains(hltA, d.wid) ){
                d3.select(this).attr('class','hltA');
            }else if( hltG.hasNode(d.wid) ){
                d3.select(this).attr('class','hltP');
            }else if( _.contains(hltA1, d.wid) ){
                d3.select(this).attr('class','hltA1');
            }else if( hltG1.hasNode(d.wid) ){
                d3.select(this).attr('class','hltP1');
            }else{
                d3.select(this).attr('class','');
            };
        });
        //all edges color
        d3.selectAll(".edge").each(function(d){
            if( hltG.hasEdge(d.source.wid,d.target.wid) ){
                d3.select(this).attr('class','edge hltE');
            }else if( hltG1.hasEdge(d.source.wid,d.target.wid) ){
                d3.select(this).attr('class','edge hltE1');
            }else{
                d3.select(this).attr('class','edge');
            };
        });
    };

    //handle text
    d3.selectAll(".gnode").selectAll("text").each(function(d){
        if ( _.contains(hltQ, d.wid)  ){
            d3.select(this).attr('class','txQ');
        }else if( _.contains(hltA, d.wid) || hltG.hasNode(d.wid) ){
            d3.select(this).attr('class','hlttx');
        }else{
            d3.select(this).attr('class','');
        };
    });


    //change title color
    TITLECOLOR_CHANGE();
};

// zooming to multiple nodes so that the nodes fill up the screen.nodes is a list
function ZoomToNodes(nodes){
    //if (  SIMULATION.alphaTarget()==0 ){
    var begin_zoom = function(){

        var obj_nodes = d3.selectAll('.gnode').filter(function(d){return _.contains(nodes,d.wid);});
        obj_nodes = obj_nodes.data();
        if(obj_nodes.length == 1){
            var k = 1;
            var x=obj_nodes[0].x
            var y=obj_nodes[0].y
            if(w<=750){ // mobel
                var movew = w/2;
                var moveh = 75+0.335*h;
            }else{  // PC
                if( d3.select('#info_panel').style('display')=="none" ){
                    var movew = w/2;
                    var moveh = h/2;
                }else{
                    var movew = (w+Width_infoPanel)/2;
                    var moveh = h/2;
                };
            };

        }else{
            var max_x=d3.max(obj_nodes,function(d){return d.x});
            var max_y=d3.max(obj_nodes,function(d){return d.y});
            var min_x=d3.min(obj_nodes,function(d){return d.x});
            var min_y=d3.min(obj_nodes,function(d){return d.y});
            var x = (max_x+min_x)/2;
            var y = (max_y+min_y)/2;
            if(w<=750){
                var movew = w/2;
                var moveh = 75+0.335*h;
                var kx = 0.8*w/(max_x-min_x+4*maxNodeRadius);
                var ky = 0.8*(0.67*h-150)/(max_y-min_y+4*maxNodeRadius);
                var k = Math.min(kx,ky);
            }else{
                if ( d3.select('#info_panel').style('display')=="none" ){
                    var kx = 0.7*w/(max_x-min_x+4*maxNodeRadius);
                    var movew = w/2;
                    var moveh = h/2;
                }else{
                    var kx = 0.7*(w-Width_infoPanel)/(max_x-min_x+4*maxNodeRadius);
                    var movew = (w+Width_infoPanel)/2;
                    var moveh = h/2;
                };
                var ky = 0.7*h/(max_y-min_y+4*maxNodeRadius);
                var k = Math.min(kx,ky);
            };

        };
        function transform(){
            return d3.zoomIdentity
                     .translate(movew, moveh )
                     .scale(k)
                     .translate(-x,-y);
        };
        if(obj_nodes.length >= 1){
            SVG.transition('zoom').duration(1000).call(User_Zoom.transform, transform);
        };

    };
    if ( SIMULATION.alphaTarget()==0 ){
        begin_zoom();
    }else{
        setTimeout(begin_zoom,1200);
    };
        //SIMULATION.on('tick.zoom',null);

    /*}else{
        SIMULATION.on('tick.zoom', function(){
            var maxv = d3.max(d3.selectAll('.gnode circle').data(),function(d){return Math.sqrt(d.vx*d.vx+d.vy*d.vy);});
            if (  SIMULATION.alphaTarget()==0 && maxv<0.1 ){
                ZoomToNodes(nodes);
            };
        });
    };*/
};

// Back to force layout
function RedoBack(){
    //resume color
    d3.selectAll(".gnode").selectAll("circle").transition().style("opacity","1");
    d3.selectAll(".gnode").selectAll("text").transition().style("opacity","1");
    d3.selectAll(".edge").transition().style("opacity","1");
    // remove neighbor track
    d3.selectAll("circle.neighbor_track").remove();
    //add force
    SIMULATION.force("link").strength(function(d) {
        return 1 / Math.min(d.source.n, d.target.n);
    });
    SIMULATION.force("center", d3.forceCenter(w / 2, h / 2));
    SIMULATION.force("charge", d3.forceManyBody());
    //remove force
    SIMULATION.force("xp",null);
    SIMULATION.force("yp",null);
    SIMULATION.alphaTarget(0.3).restart();
    //change title color
    TITLECOLOR_CHANGE();
};

//reset minhops and switchLG
function reset_hops_switcher(panel_id){
    var panel = d3.select('#'+panel_id);
    //reset minhops
    panel.select('select.minhop').each(function(d){this.value=1;});
    //reset switch
    panel.selectAll('.t-global').classed('t-global-toggle',false);
    panel.selectAll('.t-local').classed('t-local-toggle',false);
    panel.selectAll('label.switchLG input').each(function(d){this.checked=false;});
}

//get the value of minhops
function get_minhops(minhop_id){
    var selector = 'select#'+minhop_id;
    return parseInt(d3.select(selector).node().value);
};
// check swith is either local or global
function check_explore_LG(switcher){
    var selector = 'label.switchLG input#'+switcher
    if( d3.select(selector).node().checked==true ){
        return 'specific';
    }else{
        return 'general';
    };
};

// Explore either global or local graph
// Start or Previous or Next
// Query is the queried node to be highlighted
// born is the wid of the node as the bornplace.
function Explore_Nearby(LorG,start,minhops,N,query,born){
    var currentnodes = CLIENT_NODES_ids;
    if ( LorG=="specific" ){
        var subparameters = {'ipt':query,'tp':ExploreSP_distance,'minhops':minhops,'localnodes':null};
        var parameters = {'N':N,'parameters':subparameters,'generator':'get_Rel_one','start':start};
        var info = {'explorelocal': false, 'parameters':parameters,'localnodes':currentnodes};
    }else if( LorG=="general" ){
        var subparameters = {'ipt':query,'tp':ExploreG_Distance,'minhops':minhops,'localnodes':null};
        var parameters = {'N':N,'parameters':subparameters,'generator':'get_Rel_one','start':start};
        var info = {'explorelocal': false, 'parameters':parameters,'localnodes':currentnodes};
    }else{
        alert('unknown specific or general');
        throw 'unknown specific or general';
    };
    //calculate bornplace
    assert( _.contains(currentnodes,born), 'current nodes do not include born node');
    var bornnode = CLIENT_NODES.filter(function(obj){return obj["wid"]==born;})[0];
    var bornplace = {x:bornnode.x, y:bornnode.y, vx:bornnode.vx, vy: bornnode.vy};

    generator_update_graphAndPanel(info,bornplace,[query],[query]);

};




//find paths between two nodes
function findPaths_betweenNodes(LorG, start, minhops, N, node1, node2){
    if ( LorG=="specific" ){
        var subparameters = {"source":node1,"target":node2,"tp":PathSP_distance,"minhops":minhops,"localnodes":null};
        var parameters={"N":N,"parameters":subparameters,"generator":"find_paths","start":start};
        var info = {"explorelocal":false,"parameters":parameters,"localnodes":CLIENT_NODES_ids};
    }else if( LorG=="general" ){
        var subparameters = {"source":node1,"target":node2,"tp":PathG_Distance,"minhops":minhops,"localnodes":null};
        var parameters={"N":N,"parameters":subparameters,"generator":"find_paths","start":start};
        var info = {"explorelocal":false,"parameters":parameters,"localnodes":CLIENT_NODES_ids};
    }else{
        alert('unknown specific or general');
        throw 'unknown specific or general';
    };
    // calculate bornplace
    assert( _.contains(CLIENT_NODES_ids,node1) && _.contains(CLIENT_NODES_ids,node2) , 'path ends do not exist!');
    assert ( node1!=node2, 'start and end node should not be the same node!'  )
    var bornnode1=CLIENT_NODES.filter(function(obj){return obj["wid"]==node1;})[0];
    var bornnode2=CLIENT_NODES.filter(function(obj){return obj["wid"]==node2;})[0];
    var bornplace = {x:(bornnode1.x+bornnode2.x)/2, y:(bornnode1.y+bornnode2.y)/2, vx:(bornnode1.vx+bornnode2.vx)/2, vy: (bornnode1.vy+bornnode2.vy)/2 };

    generator_update_graphAndPanel(info, bornplace, [node1,node2], [node1,node2]);
};

//find paths between two clusters
function findBpaths_betweenClusters(LorG, start, N, cluster1, cluster2){
    if( _.intersection(cluster1,cluster2).length > 0 ){
        throw 'two clusters are overlapping.'
    };
    if ( LorG=='specific' ){
        var subparameters = {'cluster1':cluster1, 'cluster2':cluster2, 'tp':PathSP_distance, 'localnodes':null};
        var parameters = { 'N':N, 'parameters':subparameters, 'generator': 'find_paths_clusters','start':start  };
        var info = {'explorelocal':false, 'parameters':parameters, 'localnodes':CLIENT_NODES_ids};
    }else if( LorG=="general" ){
        var subparameters = {'cluster1':cluster1, 'cluster2':cluster2, 'tp':PathG_Distance, 'localnodes':null};
        var parameters = { 'N':N, 'parameters':subparameters, 'generator': 'find_paths_clusters','start':start  };
        var info = {'explorelocal':false, 'parameters':parameters, 'localnodes':CLIENT_NODES_ids};
    }else{
        alert('unknown specific or general');
        throw 'unknown specific or general';
    };
    var bornnode1=CLIENT_NODES.filter(function(obj){return obj["wid"]==cluster1[0];})[0];
    var bornnode2=CLIENT_NODES.filter(function(obj){return obj["wid"]==cluster2[0];})[0];
    var bornplace = {x:(bornnode1.x+bornnode2.x)/2, y:(bornnode1.y+bornnode2.y)/2, vx:(bornnode1.vx+bornnode2.vx)/2, vy: (bornnode1.vy+bornnode2.vy)/2 };

    generator_update_graphAndPanel(info, bornplace, [],[]);
};


// Access generator url, update force graph and information panel
// info for the generator server
// bornplace is the x,y,vx,vy for the newly added nodes
// znodes are the nodes should be zoomed
// queries are the query nodes
function generator_update_graphAndPanel(info,bornplace,znodes,queries){
    d3.select('div#info_panel div.info-display').selectAll('div.row').remove();
    Loading_Spinner.spin(d3.select('.info-display').node());
    d3.selectAll('.gnode,#mainSearchBox,#func-nav,#point,#point_show_results,#line,#line_show_results,#cluster,#info_panel').style("pointer-events", "none");
    d3.select('#pleasewait').style('display','block');

    d3.json('/generator/'+JSON.stringify(info),function(error,data){
        if(data.AddNew==true){
            SHOW_UPDATE_FORCE(data,bornplace); //add new node and update the graph displayed
            node_left_click_on();
        };

        var hlpath=[];
        var hlpath1 = [];
        var z_nodes =znodes.slice();
        data.paths.forEach(function(d,i){
            if (i==0){
                hlpath.push(d.ids);
            }else{
                hlpath1.push(d.ids);
            };
            z_nodes = _.union(z_nodes,d.ids);
        });
        var highlights={'nodes':queries,'paths':hlpath,'paths1':hlpath1}; //highlight nodes and paths
        highlight_nodespaths(highlights);
        ZoomToNodes(z_nodes); // zoom to the node
        //update the information panel here
        update_informationPanel(data.paths,data.position);
    });
};

// paths are the information to be updated on the information panel
// position is the position of the current paths
function update_informationPanel(paths,position){

    Loading_Spinner.stop();
    d3.selectAll('.gnode,#mainSearchBox,#func-nav,#point,#point_show_results,#line,#line_show_results,#cluster,#info_panel').style("pointer-events", null);
    d3.select('#pleasewait').style('display','none');

    var inforow = d3.select('div#info_panel div.info-display')
                    .selectAll('div.row')
                    .data(paths)
                    .enter()
                    .append('div')
                    .attr('class',function(d,i){
                        if( i==0 ){
                            return 'row clicked';
                        }else{
                            return 'row';
                        };
                    });
    inforow.append('p').attr('class','list')
                       .text(function(d,i){
                           return parseInt(i)+position;
                       });

    if ( d3.select('#point_show_results').style('display')=="block" ){
        inforow.append('p').attr('class','infoHead')
                         .text(function(d,i){
                             return d.labels.slice(-1)[0];
                         });
        inforow.append('p').attr('class','infoDetail')
                         .text(function(d,i){
                             return d.labels.join(' --> ');
                         });
    }else{
        var pathinfo = inforow.append('p').attr('class','infoHead');
        pathinfo.append('span').text(function(d){return d.labels[0];});
        pathinfo.append('span').style('color','#839192').style('font-size', '12px')
                .text(function(d){
                    if ( d.labels.slice(1,-1).length==0 ){
                        return ' --> ';
                    }else{
                        return ' --> '+d.labels.slice(1,-1).join(' --> ')+' --> ';
                    };
                });

        pathinfo.append('span').text(function(d){return d.labels.slice(-1)[0];});

    };


    //information clickable
    d3.select('div#info_panel div.info-display').selectAll('div.row').on('click',function(d,i){
        d3.selectAll('div.row').classed('clicked',false);
        d3.select(this).classed('clicked',true);
        var hltP1=[];
        var hltQ = [];
        if ( d3.select('#point').style('display')=='block' ){
            hltQ.push(d.ids[0]);
        }else if( d3.select('#line').style('display')=='block' ){
            hltQ.push(d.ids[0]);
            hltQ.push(d.ids.slice(-1)[0]);
        };
        paths.forEach(function(p,pi){
            if(pi!=i){
                hltP1.push(p.ids);
            };
        });
        var highlights={'nodes':hltQ,'paths':[d.ids],'paths1':hltP1};
        highlight_nodespaths(highlights);
        ZoomToNodes(d.ids);
    });
};


// get the setting for cluster algorithm
function get_clusterSetting(){
    //alert if there is only one node.
    if( CLIENT_NODES_ids.length<=1 ){
        alert('The number of nodes should be more than one for clustering');
        throw 'not enough nodes to be clustered';
    };

    if( d3.select("#clusterMethod").node().value=='normalized'){
        var method = 'normalized';
        var parameter = parseInt( d3.select('div#clusterMethod1Setting input').node().value );
        // if the number of clusters is too large
        if ( parameter>= CLIENT_NODES_ids.length ){
            //parameter = CLIENT_NODES_ids.length-1;
            //d3.select('div#clusterMethod1Setting input').node().value = CLIENT_NODES_ids.length-1;
            alert('The number of clusters should be less than the number of nodes');
            throw 'The number of clusters should be less than the number of nodes';
        };
    }else if(d3.select("#clusterMethod").node().value=='mcl'){
        var method = 'mcl';
        var parameter = parseInt( d3.select('div#clusterMethod2Setting input').node().value );
    }else{
        alert('please select clustering method');
        throw 'please select clustering method';
    };
    return [method,parameter];
};

//generate clusters
function generate_Clusters(){
    // zoom all

    var setting = get_clusterSetting();
    ZoomToNodes(CLIENT_NODES_ids);
    if (setting[0] == 'normalized'){
        var info = {'nodes':CLIENT_NODES_ids,'method':'normalized','weight':Kernal_Weight,'k':setting[1],'distance':Type_distance }
    }else{
        var info = {'nodes':CLIENT_NODES_ids,'method':'mcl','weight':Kernal_Weight,'r':setting[1],'distance':Type_distance }
    };
    d3.json('/generateClusters/'+JSON.stringify(info),function(error,data){
        var clusters = data
        //---------Coler_Cluster!!!!!!!!!!!!!!
        var colors = Colorized_Clusters(clusters);
        // show the number of clusters!!!!
        d3.select('#cluster_level_2').select('#numberOfClusters h4').text(function(){
            return 'Get '+clusters.length+' Clusters!';
        });
        // set options
        var options = d3.select('#findPath').selectAll('#clusterStartList, #clusterEndList')
                                            .selectAll('a')
                                            .data(clusters)
                                            .attr('value',function(d,i){
                                                return i;
                                            })
                                            .style('background-color',function(d,i){
                                                return colors[i];
                                            });
        options.enter()
               .append('a')
               .attr('href',"#")
               .attr('value',function(d,i){
                    return i;
               })
               .style('background-color',function(d,i){
                   return colors[i];
               })
               .attr('class','listTxt')
               /* show text of option
               .text(function(d){
                   return 'cluster centering at: '+NODE_IdToObj(d[0]).label;
               })*/;
        options.exit().remove();

        //option click on
        d3.selectAll('#clusterStartList a').on('click',function(d,i){
            //zoom to cluster
            var cl_nodes = d3.select(this).data()[0];
            ZoomToNodes(cl_nodes);

            var color = d3.select(this).style('background-color');
            var value = d3.select(this).attr('value');
            d3.select('#clusterStartSelect')
            .style('background-color',color)
            .attr('value',value);
            ChangeSelectCluster('clusterStartSelect');
         });
        d3.selectAll('#clusterEndList a').on('click',function(d,i){
            //zoom to cluster
            var cl_nodes = d3.select(this).data()[0];
            ZoomToNodes(cl_nodes);

            var value = d3.select(this).attr('value');
            var color = d3.select(this).style('background-color')
            d3.select('#clusterEndSelect')
              .style('background-color',color)
              .attr('value',value);
             ChangeSelectCluster('clusterEndSelect');
        });

    });

};

// Colorized Clusters
function Colorized_Clusters(clusters){
    var n_clusters = clusters.length
    // not more than 10 clusters
    if (n_clusters<=10) {
        var colors = ["#1f77b4", "#ff0", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#d6616b", "#bcbd22", "#17becf"];
        colors = _.shuffle(colors)
        colors = colors.slice(0,n_clusters)
    } else {
        var scaleColor_h = d3.scaleLinear().domain([0,n_clusters-1]).range([0,324]);
        var colors=[];
        for (var i=0;i<=n_clusters-1;i++){
            colors.push( d3.hsl( scaleColor_h(i), 1 , 0.5 ) );
        };
    };

    d3.selectAll('.gnode circle').each(function(d){
        for (var i=0;i<=n_clusters-1;i++){
            var cluster = clusters[i];
            var j = cluster.indexOf(d.wid);
            if ( j>=0 ){
                //var scaleColor_s = d3.scaleLinear().domain( [0, cluster.length-1] ).range([0.5,1.0]);
                d.icluster = i;
                d.color = colors[i];
                d3.select(this).style( 'fill' , colors[i] );
                break;
            };
        };
    });

    d3.selectAll(".edge").each(function(d){
        if(d.source.icluster == d.target.icluster ){
            d3.select(this).style('stroke',d.source.color);
        }else{
            d3.select(this).style('stroke',null);
        };
    });

    return colors;
};

//cancel cluster color
function cancelClusterColor(){
    d3.selectAll(".gnode circle").style('fill',null);
    d3.selectAll('.edge').style('stroke',null);
};
//cancel query highlight
function cancelQyHighlight(){
    d3.selectAll(".gnode circle").classed('hltQ',null);
    d3.selectAll(".gnode text").classed('txQ',null);
};
//cancel information highlight
function cancelInfoHighlight(){
    d3.selectAll(".gnode circle").classed('hltA hltA1 hltP hltP1',null);
    d3.selectAll(".gnode text").classed('hlttx',null);
    d3.selectAll(".edge").attr('class','edge');
};
//resume cluster color
function resumeClusterColor(){
    // nodes
    d3.selectAll(".gnode circle").style('fill',function(d){
        return d.color;
    });
    // edges
    d3.selectAll(".edge").each(function(d){
        if(d.source.icluster == d.target.icluster ){
            d3.select(this).style('stroke',d.source.color);
        }else{
            d3.select(this).style('stroke',null);
        };
    });
};