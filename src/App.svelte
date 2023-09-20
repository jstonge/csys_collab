<script>
  
  import { forceSimulation, forceManyBody, forceCenter, forceLink } from "d3-force";
  import { nodes, links } from '../data/csys_collab'
  import { scaleOrdinal } from "d3"

  import FileSaver from 'file-saver';  // needs to be installed

  let width=700;
  let height=450;

  const simulation = forceSimulation(nodes)
    .force("link", forceLink(links).id(d => d.id))
    .force("charge", forceManyBody().strength(-500))
    .force("center", forceCenter(width/2, height/2))
    .tick(5000);

  $: nodes_xy = simulation.nodes();

  let color_palette = ["#778da9", "#6a4c93", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#bb3e03", "#ae2012", "#9b2226"]
  let color_scale = scaleOrdinal([1,2,3,4,5], color_palette)

  async function downloadChart() {
    var chart = document.getElementsByTagName('svg')[0]
    console.log(chart)
    const svgString = new XMLSerializer().serializeToString(chart);
    const canvas = document.createElement("canvas"); // create <canvas> element
                
    canvas.width = chart.getAttribute('width');
    canvas.height = chart.getAttribute('height');

    // The 2D Context provides objects, methods, and properties to draw 
    // and manipulate graphics on a canvas drawing surface.
    const ctx = canvas.getContext("2d");

    const data = new Blob([svgString], {type: "image/svg+xml"});
    console.log(svgString)
    
    const url = URL.createObjectURL(data);

    const img = new Image();
    img.onload = () => {
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(function(blob) { FileSaver.saveAs(blob, "output.svg");});
    };
    img.src = url;
    }

</script>

<main>
  <div class='chart-container'>
  <svg {width} {height}>
    <g class='inner-chart'  transform="translate(-90, 30)">
      {#each links as e}
      <g class="links">
        <line
         x1={e.source.x}
         x2={e.target.x}
         y1={e.source.y}
         y2={e.target.y}
         opacity="0.2"
         stroke-width={Math.sqrt(e.value)}
         stroke="hsla(212, 10%, 53%, 1)"
         ></line>  
      </g>
      {/each}
      {#each nodes_xy as node}
      <g class="nodes">
        <circle
          cx={node.x}
          cy={node.y}
          fill={color_scale(node.group)}
          stroke="black"
          opacity=1.
          stroke-width=1
          r={10}></circle>  
        </g>
        <g>
        <text
          x={node.x}
          y={node.y}
          dx={7}
          stroke="black"
          opacity=1.
          stroke-width=0.1
          r={7}>{node.label[0]}. {node.label.split(" ")[node.label.split(" ").length-1]}</text>  
        </g>
        {/each}
      </g>
      </svg>
    </div>
    <div>
      <button on:click={downloadChart}>Download Image</button>
    </div>
  </main>

  <style>
  .nodes {
    filter: drop-shadow(3px 5px 2px rgb(0 0 0 / 0.4));
  }

  :global(*) {
    font-family: Inter;
    -moz-osx-font-smoothing: grayscale;
  }

  .chart-container {
    width: 100%;
    height: 100%;

    /* Optional */
    max-width: 700px;
    max-height: 450px;

    background: white;
    box-shadow: 1px 1px 30px "black";
    border: 1px solid grey;
    border-radius: 6px;
  }

</style>