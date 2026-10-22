"""Interactive two-sided mind map (D3) rendered inside Streamlit."""
import json
import pathlib

CDN_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
D3_FILE = pathlib.Path(__file__).parent / "static" / "d3.min.js"   # bundled copy: works offline, no CDN round-trip
try:
    D3_TAG = f"<script>{D3_FILE.read_text(encoding='utf-8')}</script>"
except OSError:
    D3_TAG = CDN_TAG                                               # fall back to the CDN if the file is missing

TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
html,body{margin:0;height:100%;font-family:system-ui,Segoe UI,sans-serif}
#wrap{position:relative;height:100%;border-radius:18px;overflow:hidden;border:1px solid #2a3050;
background:radial-gradient(circle at 50% 40%,#1b2145 0%,#0b1020 70%)}
#wrap:before{content:"";position:absolute;inset:0;background-image:radial-gradient(#ffffff14 1px,transparent 1px);background-size:26px 26px}
svg{position:absolute;inset:0;width:100%;height:100%}
#bar{position:absolute;top:12px;right:12px;display:flex;gap:8px;z-index:3}
button{background:#ffffff12;color:#e5e7eb;border:1px solid #ffffff25;border-radius:999px;padding:6px 14px;font-size:12px;cursor:pointer;backdrop-filter:blur(6px)}
button:hover{background:#ffffff26}
#tip{position:absolute;display:none;max-width:320px;background:#0b1020ee;color:#e5e7eb;border:1px solid #818cf855;border-radius:12px;padding:10px 12px;font-size:12.5px;line-height:1.45;z-index:5;pointer-events:none;box-shadow:0 8px 30px #0008}
#hint{position:absolute;left:14px;bottom:10px;color:#94a3b8;font-size:11.5px;z-index:3}
</style></head><body><div id="wrap">
<div id="bar"><button id="ex">Expand all</button><button id="co">Collapse</button><button id="rs">Reset view</button></div>
<svg id="svg"><defs>
<filter id="glow" x="-30%" y="-50%" width="160%" height="200%"><feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#000" flood-opacity=".45"/></filter>
<linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c4b5fd"/><stop offset="1" stop-color="#7dd3fc"/></linearGradient>
</defs><g id="g"></g></svg>
<div id="tip"></div><div id="hint">Scroll to zoom · drag to pan · click a topic to fold · hover for the source sentence</div></div>
<script>
const DATA=__DATA__;
const PAL=["#818cf8","#f87171","#2dd4bf","#fbbf24","#38bdf8","#f472b6","#a3e635","#fb923c"];
const wrap=document.getElementById("wrap"),tip=document.getElementById("tip");
const svg=d3.select("#svg"),g=d3.select("#g");
const collapsed=new Set();
const zoom=d3.zoom().scaleExtent([.2,3]).on("zoom",e=>g.attr("transform",e.transform));
svg.call(zoom);
const trunc=s=>s.length>22?s.slice(0,21)+"…":s;
function fit(){const W=wrap.clientWidth,H=wrap.clientHeight,b=g.node().getBBox();
 const k=Math.min(W/(b.width+100),H/(b.height+100),1.25);
 svg.transition().duration(500).call(zoom.transform,d3.zoomIdentity.translate(W/2-k*(b.x+b.width/2),H/2-k*(b.y+b.height/2)).scale(k));}
function tree(arr){
 const kids=arr.map((t,i)=>({name:t.name,tip:t.tip,ci:t.ci,topic:true,
  children:collapsed.has(t.name)?[]:t.children.map(c=>({name:c.name,tip:c.tip,ci:t.ci}))}));
 return d3.tree().nodeSize([40,250]).separation((a,b)=>a.parent===b.parent?1:1.35)(d3.hierarchy({name:DATA.root,root:true,children:kids}));
}
function draw(refit){
 const topics=DATA.topics.map((t,i)=>({...t,ci:i}));
 const half=Math.ceil(topics.length/2);
 const sides=[[tree(topics.slice(0,half)),1],[tree(topics.slice(half)),-1]];
 const nodes=[],links=[],pos={};
 sides.forEach(([T,s])=>{
  T.descendants().forEach(d=>{if(d.depth===0&&s===-1)return;
   const n={...d.data,x:s*d.y,y:d.x,depth:d.depth};nodes.push(n);if(!n.root)pos[n.name]=n;});
  T.links().forEach(l=>links.push({s:{x:s*l.source.y,y:l.source.x},t:{x:s*l.target.y,y:l.target.x},ci:l.target.data.ci,d:l.target.depth}));
 });
 g.selectAll("*").remove();
 const linkG=g.append("g"),relG=g.append("g"),nodeG=g.append("g");
 links.forEach(l=>{const m=(l.s.x+l.t.x)/2;
  linkG.append("path").attr("d",`M${l.s.x},${l.s.y}C${m},${l.s.y} ${m},${l.t.y} ${l.t.x},${l.t.y}`)
   .attr("fill","none").attr("stroke",PAL[l.ci%8]).attr("stroke-width",l.d===1?4.5:2).attr("stroke-linecap","round").attr("opacity",.8);});
 DATA.relations.forEach(([a,v,b])=>{const A=pos[a],B=pos[b];if(!A||!B)return;
  const cx=(A.x+B.x)/2-(B.y-A.y)*.25,cy=(A.y+B.y)/2+(B.x-A.x)*.25;
  relG.append("path").attr("d",`M${A.x},${A.y}Q${cx},${cy} ${B.x},${B.y}`).attr("fill","none").attr("stroke","#cbd5e1").attr("stroke-dasharray","5 4").attr("opacity",.45);
  relG.append("text").attr("x",.25*A.x+.5*cx+.25*B.x).attr("y",.25*A.y+.5*cy+.25*B.y).attr("text-anchor","middle").attr("font-size",10).attr("fill","#94a3b8").text(v);});
 nodes.forEach(n=>{
  const ng=nodeG.append("g").attr("transform",`translate(${n.x},${n.y})`).style("cursor",n.topic?"pointer":"default").style("opacity",0);
  const col=n.root?"#fff":PAL[n.ci%8];
  const t=ng.append("text").text(trunc(n.name)).attr("text-anchor","middle").attr("dy","0.35em")
   .attr("font-size",n.root?18:n.topic?14:12.5).attr("font-weight",n.root||n.topic?700:500).attr("fill",n.root||n.topic?"#0b1020":"#e5e7eb");
  const w=t.node().getComputedTextLength()+(n.root?46:30),h=n.root?50:n.topic?38:30;
  ng.insert("rect","text").attr("x",-w/2).attr("y",-h/2).attr("width",w).attr("height",h).attr("rx",h/2)
   .attr("fill",n.root?"url(#rg)":n.topic?col:col+"22").attr("stroke",col).attr("stroke-width",n.root?0:1.6)
   .attr("filter",n.root||n.topic?"url(#glow)":null);
  if(n.topic&&collapsed.has(n.name))ng.append("text").text("＋").attr("x",w/2+12).attr("dy","0.35em").attr("fill",col).attr("font-size",16).attr("font-weight",700);
  ng.on("mousemove",ev=>{if(!n.tip)return;const r=wrap.getBoundingClientRect();
    tip.style.display="block";tip.textContent=n.tip;
    tip.style.left=Math.min(ev.clientX-r.left+14,r.width-340)+"px";tip.style.top=(ev.clientY-r.top+14)+"px";})
   .on("mouseleave",()=>tip.style.display="none")
   .on("click",()=>{if(n.topic){collapsed.has(n.name)?collapsed.delete(n.name):collapsed.add(n.name);draw(false);}});
  ng.transition().delay(n.depth*120).duration(450).style("opacity",1);
 });
 if(refit)setTimeout(fit,50);
}
document.getElementById("ex").onclick=()=>{collapsed.clear();draw(true)};
document.getElementById("co").onclick=()=>{DATA.topics.forEach(t=>collapsed.add(t.name));draw(true)};
document.getElementById("rs").onclick=fit;
draw(true);
</script></body></html>"""


def render(mm) -> str:
    topics = [{
        "name": t,
        "tip": mm.topic_concepts[t].sentences[0],
        "children": [{"name": c.name, "tip": c.sentences[0]} for c in kids],
    } for t, kids in mm.topics.items()]
    data = {"root": mm.root, "topics": topics, "relations": [list(r) for r in mm.relations]}
    html = TEMPLATE.replace(CDN_TAG, D3_TAG, 1)
    return html.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
