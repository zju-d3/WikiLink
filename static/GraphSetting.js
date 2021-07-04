

var Spinner_Opts = {
  lines: 13 // The number of lines to draw
, length: 18 // The length of each line
, width: 14 // The line thickness
, radius: 42 // The radius of the inner circle
, scale: 0.5 // Scales overall size of the spinner
, corners: 1 // Corner roundness (0..1)
, color: '#000' // #rgb or #rrggbb or array of colors
, opacity: 0.25 // Opacity of the lines
, rotate: 0 // The rotation offset
, direction: 1 // 1: clockwise, -1: counterclockwise
, speed: 1 // Rounds per second
, trail: 60 // Afterglow percentage
, fps: 20 // Frames per second when using setTimeout() as a fallback for CSS
, zIndex: 2e9 // The z-index (defaults to 2000000000)
, className: 'spinner' // The CSS class to assign to the spinner
, top: '50%' // Top position relative to parent
, left: '50%' // Left position relative to parent
, shadow: false // Whether to render a shadow
, hwaccel: false // Whether to use hardware acceleration
, position: 'absolute' // Element positioning
}
var Loading_Spinner = new Spinner(Spinner_Opts).spin(d3.select('.info-display').node());


d3.selection.prototype.moveToFront = function() {
      return this.each(function(){
        this.parentNode.appendChild(this);
      });
    };
d3.selection.prototype.moveToBack = function() {
    return this.each(function() {
        var firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};

function GETRANDOMINT(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

var TITLECOLOR_CHANGE = function(){
      d3.select("div.header").select("h1").transition().duration(1000).style("color",function(){
      var r=GETRANDOMINT(0,255);
      var g=GETRANDOMINT(0,255);
      var b=GETRANDOMINT(0,255);
      return "rgb(" + [r, g, b].join(",") + ")";
  });
};


var w=window.innerWidth || document.body.clientWidth;
var h=window.innerHeight || document.body.clientHeight;
var Width_infoPanel = 360;
var maxlinkdistance=200;
var minlinkdistance=50;
var maxlinkwidth=3;
var minlinkwidth=1;
var maxNodeRadius=20;
var minNodeRadius=8;
var tran = d3.transition()
             .duration(5000)
             .ease(d3.easeLinear);

var HltNodesNumber=20;
var POSITIONFORCE_STRENGTH=0.8;
var N_SearchButton=10;
var N_forPath = 5;
var Type_distance = 'R_n_HM';
console.log('Type_distance');
console.log(Type_distance);
var ExploreG_Distance = 'R_n_HM';
console.log('ExploreG_Distance')
console.log(ExploreG_Distance)
var ExploreSP_distance = 'R_r_GM';
console.log('ExploreSP_distance');
console.log(ExploreSP_distance);
var PathG_Distance = 'R_r_GM';
console.log('PathG_Distance')
console.log(PathG_Distance)
var PathSP_distance = 'R_n_GM';
console.log('PathSP_distance');
console.log(PathSP_distance);
var Kernal_Weight = 'weight';
var HltPathColor = '#FF6800';
var NodeColor = '#3498DB';
var EdgeColor = '#aaa';
var FOCUSING_NODE = -1;
var FOCUSING_CLUSTER = -1;
//add svg
User_Zoom = d3.zoom()
              .scaleExtent([1/4,4])
              .on("zoom",zoomed);
SVG = d3.select('body').append('svg').attr('id',"Mainback").attr('width',w) .attr('height',h).call(User_Zoom);
function SVG_change_size(){
    w=window.innerWidth || document.body.clientWidth;
    h=window.innerHeight || document.body.clientHeight;
    SVG.attr('width',w) .attr('height',h);
    BACKLAYER.attr("width", w).attr("height", h);
};

//add  main canvas
GRAPH = SVG.append("g")
           .attr("id","MainGraph");


BACKLAYER = SVG.insert("rect",":first-child")
                  .attr("id","Backlayer")
                  .attr("width", w)
                  .attr("height", h);

//set nodes and edges and simulation
CLIENT_NODES=[];
CLIENT_NODES_ids=[];
CLIENT_EDGES=[];
//define SIMULATION
SIMULATION = d3.forceSimulation()
               .force("link",d3.forceLink().id(function id(d){return d.wid;}).links(CLIENT_EDGES)) //add spring
               .force("charge", d3.forceManyBody().strength(-100)) //repel each other
               .force("center", d3.forceCenter(w / 2, h / 2)) // force to center
               .nodes(CLIENT_NODES);
// tick on
  TICK = function(){
      if( SIMULATION.alpha()>=0.49 ){
          SIMULATION.alphaTarget(0);
      };

      GRAPH.selectAll(".edge").attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

      GRAPH.selectAll(".gnode").attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

      GRAPH.selectAll(".edgelabel").attr("x", function(d) { return (d.source.x+d.target.x)/2; })
      .attr("y", function(d) { return (d.source.y+d.target.y)/2; });
  };

  SIMULATION.on("tick",TICK);

// drag behavior

  function dragstarted(d) {
       if (!d3.event.active) SIMULATION.alphaTarget(0.3).restart();
       d.fx = d.x;
       d.fy = d.y;
  };

  function dragged(d) {
       d.fx = d3.event.x;
       d.fy = d3.event.y;
  };

  function dragended(d) {
       if (!d3.event.active) SIMULATION.alphaTarget(0);
       d.fx = null;
       d.fy = null;
  };
  function zoomed() {

  GRAPH.attr("transform", d3.event.transform);
};


///////////////////////////////////////////// common - action setting//////////////////////////////////////////////////////
//alert fisttime visit
d3.json('/checkFirstTimevisit',function(error,data){
    if(data==true){
        alert('Tips: 1. Play it just like Google map;\n        2. You can also select node by clicking it instead of typing in the searchbox'+
       '\n \n                      Play it hard and Have fun, Thanks!')
    };
});





// hide information panel
function Hide_InfoPanel(){
    document.getElementById("info_panel").style.display = "none";
    cancelInfoHighlight();
};

function resetMinihop_Explore(){
    d3.select('#minhop_point').each(function(d){this.value=1;});
};