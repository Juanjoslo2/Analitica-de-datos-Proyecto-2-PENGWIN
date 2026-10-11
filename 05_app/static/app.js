const $=s=>document.querySelector(s);
let T=null,R=null,Rmalla=null,pend=null;
const NOM={SA:"Sacro",LI:"Coxal izq.",RI:"Coxal der."};

function estado(msg,frac){$("#estado").textContent=msg||"";if(frac!=null)$("#barra").style.width=(100*frac).toFixed(0)+"%";}

// cualquier 401 (sesión vencida) lleva al ingreso
const _f=window.fetch;window.fetch=async(...a)=>{const r=await _f(...a);if(r.status===401){location.href="login";}return r;};
$("#salir").onclick=async()=>{await fetch("api/logout",{method:"POST"});location.href="login";};
const EXT=[".mha",".nii",".nii.gz",".nrrd"];
fetch("api/info").then(r=>r.json()).then(i=>{
  $("#modeloInfo").textContent=`modelo ${i.modelo} · ${i.dispositivo}`;$("#quien").textContent=i.usuario;
  const e=$("#ejemplos");
  if(i.ejemplos.length){const s=document.createElement("span");s.className="nota";s.style.margin="0";s.textContent="Casos de prueba:";e.append(s);}
  i.ejemplos.forEach(x=>{const b=document.createElement("button");b.textContent=x.titulo;b.onclick=()=>abrir(x.id);e.append(b);});
  // enlaces directos: ?caso=075&v=v3 abre ese caso en esa pestaña
  const q=new URLSearchParams(location.search);
  if(q.get("caso")){const v=q.get("v");if(v){const b=document.querySelector(`.pestanas button[data-v="${v}"]`);if(b)b.click();}abrir(q.get("caso"));}
});

document.querySelectorAll(".pestanas button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".pestanas button").forEach(o=>o.classList.toggle("activo",o===b));
  document.querySelectorAll(".vista").forEach(v=>v.classList.toggle("on",v.id===b.dataset.v));
  if(b.dataset.v==="v3")malla();
});

$("#fVol").onchange=async ev=>{
  const f=ev.target.files[0];if(!f)return;ev.target.value="";
  if(!EXT.some(e=>f.name.toLowerCase().endsWith(e))){estado(`"${f.name}" no es un volumen admitido: use ${EXT.join(", ")}`,0);return;}
  const fd=new FormData();fd.append("nombre",f.name);fd.append("tamano",f.size);
  let r=await fetch("api/subida",{method:"POST",body:fd});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  const {id}=await r.json();const P=8*1024*1024;
  for(let o=0;o<f.size;o+=P){
    for(let intento=0;;intento++){
      try{r=await fetch("api/subida/"+id,{method:"PUT",body:f.slice(o,o+P)});if(r.ok)break;}catch(e){}
      if(intento>=3){estado("La subida falló; revise la conexión y vuelva a intentar.",0);return;}
    }
    estado(`Subiendo ${(Math.min(o+P,f.size)/1048576).toFixed(0)} / ${(f.size/1048576).toFixed(0)} MB`,0.3*Math.min(o+P,f.size)/f.size);
  }
  estado("Verificando el archivo…",0.3);
  r=await fetch(`api/subida/${id}/fin`,{method:"POST"});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  esperar(id);
};

function esperar(id){
  const t=setInterval(async()=>{
    const s=await (await fetch("api/trabajo/"+id)).json();
    if(s.estado==="listo"){clearInterval(t);estado("Listo",1);abrir(id);}
    else if(s.estado==="error"){clearInterval(t);estado(s.mensaje,0);}
    else estado(s.mensaje,0.3+0.7*(s.progreso||0));
  },1500);
}

$("#fImg").onchange=async ev=>{
  const f=ev.target.files[0];if(!f)return;ev.target.value="";estado("Procesando el corte…",0.5);
  const fd=new FormData();fd.append("archivo",f);fd.append("mm_px",$("#mmpx").value);
  const r=await fetch("api/imagen",{method:"POST",body:fd});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  const j=await r.json();$("#imgOut").src="data:image/png;base64,"+j.png;
  $("#resImg").classList.remove("oculto");$("#resImg").scrollIntoView({behavior:"smooth"});estado("Listo",1);
};

async function abrir(id){
  T=id;Rmalla=null;R=await (await fetch(`api/trabajo/${id}/resumen`)).json();
  $("#resultado").classList.remove("oculto");
  const sec=R.fragmentos.filter(f=>!f.principal).length;
  const sp=R.spacing_zyx_mm.map(v=>v.toFixed(2)).join(" × ");
  $("#meta").innerHTML=[
    ["Caso",R.titulo||"volumen subido"],["Cortes",R.cortes],["Spacing z × y × x (mm)",sp],
    ["Fragmentos",`${R.fragmentos.length} (${sec} secundarios)`],
    ["Modelo por corte",`${R.ms_por_corte} ms · ${R.dispositivo}`],["Pipeline completo",`${R.segundos_total} s`]
  ].map(([a,b])=>`<div>${a}<b>${b}</b></div>`).join("");
  $("#mipAx").src=`api/trabajo/${id}/mip/axial`;$("#mipCor").src=`api/trabajo/${id}/mip/coronal`;
  $("#mipSag").src=`api/trabajo/${id}/mip/sagital`;$("#giro").value=0;$("#mipGiro").src=`api/trabajo/${id}/mip/0`;
  for(let k=0;k<24;k++){new Image().src=`api/trabajo/${id}/mip/${k}`;}
  const zs=R.cortes_secundarios.length?R.cortes_secundarios:R.cortes_con_hueso;$("#z").max=R.cortes-1;
  $("#z").value=zs.length?zs[Math.floor(zs.length/2)]:Math.floor(R.cortes/2);
  corte();tabla();
  if(document.querySelector(".pestanas button.activo").dataset.v==="v3")malla();
  $("#resultado").scrollIntoView({behavior:"smooth"});
}

$("#giro").oninput=e=>$("#mipGiro").src=`api/trabajo/${T}/mip/${e.target.value}`;
function corte(){
  const z=$("#z").value,q=`mascaras=${+$("#cM").checked}&cajas=${+$("#cC").checked}&distancias=${+$("#cD").checked}`;
  clearTimeout(pend);pend=setTimeout(()=>$("#corteImg").src=`api/trabajo/${T}/corte/${z}?${q}`,60);
}
["#z","#cM","#cC","#cD"].forEach(s=>$(s).oninput=corte);
$("#zMenos").onclick=()=>{$("#z").value=+$("#z").value-1;corte();};
$("#zMas").onclick=()=>{$("#z").value=+$("#z").value+1;corte();};
$("#zSec").onclick=()=>{
  const z=+$("#z").value,c=R.cortes_secundarios||[];
  const sig=c.find(v=>v>z+2)??c[0];if(sig!=null){$("#z").value=sig;corte();}
};

async function malla(){
  if(!T)return;if(!Rmalla)Rmalla=await (await fetch(`api/trabajo/${T}/malla`)).json();
  const tr=Rmalla.trazas.map(m=>({type:"mesh3d",x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,color:m.color,
    opacity:m.principal?0.5:1,name:m.etiqueta,hovertemplate:m.etiqueta+"<extra></extra>",
    lighting:{ambient:0.5,diffuse:0.8,specular:0.2,roughness:0.6}}));
  const s=Rmalla.trazas.filter(m=>!m.principal);
  tr.push({type:"scatter3d",mode:"text+markers",x:s.map(m=>m.centro[0]),y:s.map(m=>m.centro[1]),z:s.map(m=>m.centro[2]),
    text:s.map(m=>m.etiqueta),textposition:"top center",textfont:{color:"#ffffff",size:12},
    marker:{size:3,color:s.map(m=>m.color)},hoverinfo:"text",showlegend:false});
  const ejes={showbackground:false,color:"#9aa1ad",gridcolor:"#2a2f3a",zerolinecolor:"#2a2f3a"};
  Plotly.react("malla",tr,{paper_bgcolor:"#171a21",plot_bgcolor:"#171a21",font:{color:"#e6e8ec"},
    margin:{l:0,r:0,t:10,b:0},showlegend:false,
    scene:{aspectmode:"data",xaxis:{...ejes,title:"x (mm) der.→izq."},
      yaxis:{...ejes,title:"y (mm) ant.→post."},zaxis:{...ejes,title:"z (mm) inf.→sup."},
      camera:{eye:{x:0,y:-1.9,z:0.4},up:{x:0,y:0,z:1}}}},{responsive:true,displaylogo:false});
}

function tabla(){
  $("#tabla").innerHTML=R.fragmentos.map(f=>`<tr><td><i class="pastilla" style="background:${f.color}"></i>${f.region}-${f.k}${f.principal?" (principal)":""}</td>
    <td>${NOM[f.region]}</td><td>${f.volumen_cm3.toFixed(1)} cm³</td>
    <td>${f.principal?"—":(f.dist_mm!=null?f.dist_mm.toFixed(2)+" mm":"—")}</td>
    <td>${f.en_contacto==null?"—":(f.en_contacto?"sí":"no")}</td></tr>`).join("");
  const g=R.separacion_gt;$("#gtBloque").classList.toggle("oculto",!g);
  if(!g)return;
  const m=R.metricas_gt||{},n=v=>v==null?"—":v.toFixed(3);
  $("#gtMet").innerHTML=[["Dice por fragmento",n(m.dice_fragmento)],["IoU por fragmento",n(m.iou_fragmento)],
    ["Secundarios recuperados",m["recuperados_secundarios_%"]==null?"—":m["recuperados_secundarios_%"].toFixed(0)+" %"],
    ["Error medio de distancia",m.mae_distancia_mm==null?"—":m.mae_distancia_mm.toFixed(2)+" mm"]]
    .map(([a,b])=>`<div>${a}<b>${b}</b></div>`).join("");
  $("#tablaGt").innerHTML=g.map(r=>`<tr><td>${r.region}-${(r.gt_label-1)%10+1}</td><td>${r.volumen_cm3.toFixed(1)} cm³</td>
    <td>${r.dist_gt_mm.toFixed(2)} mm${r.contacto_gt?" (contacto)":""}</td>
    <td>${r.medido?r.dist_pred_mm.toFixed(2)+" mm":"no recuperado"}</td>
    <td>${r.medido?(r.error_mm>=0?"+":"")+r.error_mm.toFixed(2)+" mm":"—"}</td></tr>`).join("");
}
