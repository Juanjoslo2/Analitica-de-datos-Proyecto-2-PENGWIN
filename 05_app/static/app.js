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

// ---------------------------------------------------------------- entrada: volumen, cortes o carpeta
$("#fVol").onchange=ev=>{const f=[...ev.target.files];ev.target.value="";entrada(f.map(x=>({file:x,nombre:x.name})));};
$("#fCortes").onchange=ev=>{const f=[...ev.target.files];ev.target.value="";entrada(f.map(x=>({file:x,nombre:x.name})));};
$("#fCarpeta").onchange=ev=>{const f=[...ev.target.files];ev.target.value="";
  entrada(f.map(x=>({file:x,nombre:x.webkitRelativePath||x.name})));};
if(!("webkitdirectory" in document.createElement("input")))$("#lCarpeta").classList.add("oculto");

// arrastrar y soltar: archivos sueltos o carpetas enteras (se recorren con webkitGetAsEntry)
const zona=$("#entrada");
["dragenter","dragover"].forEach(t=>zona.addEventListener(t,e=>{e.preventDefault();zona.classList.add("arrastrando");}));
["dragleave","drop"].forEach(t=>zona.addEventListener(t,e=>{e.preventDefault();zona.classList.remove("arrastrando");}));
zona.addEventListener("drop",async e=>{
  const items=[...(e.dataTransfer.items||[])].map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
  if(!items.length){entrada([...e.dataTransfer.files].map(x=>({file:x,nombre:x.name})));return;}
  estado("Leyendo lo que soltó…",0);
  const lista=[];for(const it of items)await recorrer(it,"",lista);
  entrada(lista);
});
async function recorrer(ent,ruta,lista){
  if(ent.isFile){const file=await new Promise((ok,mal)=>ent.file(ok,mal));lista.push({file,nombre:ruta+file.name});return;}
  const lector=ent.createReader();
  for(;;){const lote=await new Promise((ok,mal)=>lector.readEntries(ok,mal));if(!lote.length)break;   // de a 100
    for(const h of lote)await recorrer(h,ruta+ent.name+"/",lista);}
}

// tipo por los primeros bytes, como lo hace el servidor (cortes.tipo_de)
async function tipoDe(f){
  const b=new Uint8Array(await f.slice(0,132).arrayBuffer());
  if(b[0]===0x89&&b[1]===0x50&&b[2]===0x4e&&b[3]===0x47)return "png";
  if(b[0]===0xff&&b[1]===0xd8&&b[2]===0xff)return "jpeg";
  if(b.length>=132&&b[128]===0x44&&b[129]===0x49&&b[130]===0x43&&b[131]===0x4d)return "dicom";
  return null;
}
const esVolumen=n=>EXT.some(e=>n.toLowerCase().endsWith(e))||n.toLowerCase().endsWith(".zip");

async function entrada(lista){
  if(!lista.length)return;
  if(lista.length===1&&esVolumen(lista[0].file.name)){subirArchivo(lista[0].file,lista[0].file.name);return;}
  if(lista.some(a=>esVolumen(a.file.name))){estado("Suba el volumen (o el .zip) solo, no junto con otros archivos.",0);return;}
  estado(`Revisando ${lista.length} archivo(s)…`,0);
  const tipos=await Promise.all(lista.map(a=>tipoDe(a.file)));
  const utiles=lista.filter((_,k)=>tipos[k]),fuera=lista.length-utiles.length;
  if(!utiles.length){estado(lista.length===1?`"${lista[0].file.name}" no es un volumen ni un corte PNG, JPG o DICOM.`
    :"Ninguno de los archivos es un corte PNG, JPG o DICOM.",0);return;}
  if(utiles.length===1){corteSuelto(utiles[0].file);return;}
  // varios cortes: se juntan en un solo archivo (.lote) y suben por la misma vía que un volumen
  const lote=empacar(utiles);
  subirArchivo(lote,"cortes.lote",`${utiles.length} cortes`+(fuera?` (${fuera} archivos que no son cortes quedaron fuera)`:""));
}

// formato del lote (cortes.leer_lote): "PWLOTE1\n" + u32 largo del índice + índice JSON + los archivos seguidos.
// El Blob no copia nada: cada trozo de 8 MB se lee de los archivos originales al subirlo.
function empacar(utiles){
  const idx=new TextEncoder().encode(JSON.stringify({archivos:utiles.map(a=>({nombre:a.nombre,tam:a.file.size}))}));
  const cab=new Uint8Array(12);cab.set(new TextEncoder().encode("PWLOTE1\n"));new DataView(cab.buffer).setUint32(8,idx.length,true);
  return new Blob([cab,idx,...utiles.map(a=>a.file)]);
}

function opcionesCortes(fd){
  for(const [k,s] of [["mm_px","#mmpx"],["mm_corte","#mmcorte"],["nivel","#nivel"],["ancho","#ancho"]])fd.append(k,$(s).value.trim());
}

async function subirArchivo(f,nombre,que){
  const fd=new FormData();fd.append("nombre",nombre);fd.append("tamano",f.size);opcionesCortes(fd);
  let r=await fetch("api/subida",{method:"POST",body:fd});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  const {id,trozo:P}=await r.json();
  try{await subir(f,id,P,que,nombre);}catch(e){estado(e.fatal?`Archivo rechazado: ${e.message}`:"La subida falló; revise la conexión y vuelva a intentar.",0);return;}
  estado("Verificando el archivo…",0.3);
  r=await fetch(`api/subida/${id}/fin`,{method:"POST"});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  esperar(id);
}

// Subida: trozos de 8 MB comprimidos con gzip y 4 a la vez, cada uno por su WebSocket (= su propia
// conexión TCP; por HTTP/2 todo iría en una sola y a 0,3 MB/s). Sin WebSocket, PUT de a uno.
const PARALELO=4,GZIP=typeof CompressionStream!=="undefined";
const mb=b=>(b/1048576).toFixed(0);
const dura=s=>s<90?`${Math.max(1,Math.round(s))} s`:`${Math.round(s/60)} min`;

async function subir(f,id,P,que,nombre){
  const N=Math.ceil(f.size/P),comprimir=GZIP&&!/\.(gz|zip)$/i.test(nombre);
  let sig=0,listo=0,fallo=null;const t0=performance.now(),abiertos=[];
  const ws0=await abrirWS(id).catch(()=>null);
  async function canal(k){
    let ws=k===0?ws0:await abrirWS(id).catch(()=>null);
    if(ws)abiertos.push(ws);else if(k>0)return;              // si un canal extra no abre, los demás se reparten sus trozos
    while(!fallo&&sig<N){
      const t=await trozo(f,sig++,P,comprimir);
      for(let intento=0;!fallo;intento++){
        try{
          if(!ws0){await enviarPUT(id,t);break;}
          if(!ws||ws.readyState!==WebSocket.OPEN){ws=await abrirWS(id);abiertos.push(ws);}
          await enviarWS(ws,t);break;
        }catch(e){
          if(e.fatal||intento>=3)throw e;
          if(ws)ws.close();ws=null;await new Promise(z=>setTimeout(z,1500*(intento+1)));
        }
      }
      listo+=t.n;
      const s=(performance.now()-t0)/1000,v=listo/s;
      estado(`Subiendo ${que?que+" · ":""}${mb(listo)} / ${mb(f.size)} MB · ${(v/1048576).toFixed(1)} MB/s · faltan ~${dura((f.size-listo)/v)}`,0.3*listo/f.size);
    }
  }
  await Promise.all(Array.from({length:ws0?Math.min(PARALELO,N):1},(_,k)=>canal(k).catch(e=>{fallo=fallo||e;})));
  abiertos.forEach(w=>w.close());
  if(fallo)throw fallo;
}

async function trozo(f,i,P,comprimir){
  const b=f.slice(i*P,Math.min((i+1)*P,f.size));
  if(comprimir){
    const c=new Uint8Array(await new Response(b.stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer());
    if(c.length<0.95*b.size)return {i,n:b.size,z:1,datos:c};   // ya comprimido (p. ej. .mha con CompressedData): va tal cual
  }
  return {i,n:b.size,z:0,datos:b};
}

function abrirWS(id){
  const u=new URL(`api/subida/${id}/ws`,location.href);u.protocol=u.protocol==="https:"?"wss:":"ws:";
  return new Promise((ok,mal)=>{
    const ws=new WebSocket(u);ws.binaryType="arraybuffer";
    const t=setTimeout(()=>{ws.close();mal(new Error("el WebSocket no abrió"));},15000);
    ws.onopen=()=>{clearTimeout(t);ok(ws);};
    ws.onerror=()=>{clearTimeout(t);mal(new Error("el WebSocket no abrió"));};
  });
}

function enviarWS(ws,t){
  return new Promise((ok,mal)=>{
    const cab=new Uint8Array(9),dv=new DataView(cab.buffer);
    dv.setUint32(0,t.i,true);dv.setUint32(4,t.n,true);cab[8]=t.z;
    const reloj=setTimeout(()=>ws.close(),180000);            // trozo atascado: se cierra y se reintenta
    ws.onmessage=e=>{clearTimeout(reloj);const r=JSON.parse(e.data);
      if(r.error){const x=new Error(r.error);x.fatal=true;mal(x);}else ok(r);};
    ws.onclose=()=>{clearTimeout(reloj);mal(new Error("conexión cerrada"));};
    ws.send(new Blob([cab,t.datos]));
  });
}

async function enviarPUT(id,t){
  const r=await fetch(`api/subida/${id}?i=${t.i}&n=${t.n}&z=${t.z}`,{method:"PUT",body:t.datos});
  if(r.ok)return r.json();
  const x=new Error(r.status===400?(await r.json()).detail:`HTTP ${r.status}`);x.fatal=r.status===400||r.status===404;throw x;
}

function esperar(id){
  const t=setInterval(async()=>{
    const s=await (await fetch("api/trabajo/"+id)).json();
    if(s.estado==="listo"){clearInterval(t);estado("Listo",1);abrir(id);}
    else if(s.estado==="error"){clearInterval(t);estado(s.mensaje,0);}
    else estado(s.mensaje,0.3+0.7*(s.progreso||0));
  },1500);
}

async function corteSuelto(f){
  estado("Procesando el corte…",0.5);
  const fd=new FormData();fd.append("archivo",f);fd.append("mm_px",$("#mmpx").value.trim());
  fd.append("nivel",$("#nivel").value.trim());fd.append("ancho",$("#ancho").value.trim());
  const r=await fetch("api/imagen",{method:"POST",body:fd});
  if(!r.ok){estado((await r.json()).detail,0);return;}
  const j=await r.json();$("#imgOut").src="data:image/png;base64,"+j.png;$("#imgNota").textContent=j.nota||"";
  $("#resImg").classList.remove("oculto");$("#resImg").scrollIntoView({behavior:"smooth"});estado("Listo",1);
}

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
  reconstruccion(id);
  $("#mipAx").src=`api/trabajo/${id}/mip/axial`;$("#mipCor").src=`api/trabajo/${id}/mip/coronal`;
  $("#mipSag").src=`api/trabajo/${id}/mip/sagital`;$("#giro").value=0;$("#mipGiro").src=`api/trabajo/${id}/mip/0`;
  for(let k=0;k<24;k++){new Image().src=`api/trabajo/${id}/mip/${k}`;}
  const zs=R.cortes_secundarios.length?R.cortes_secundarios:R.cortes_con_hueso;$("#z").max=R.cortes-1;
  $("#z").value=zs.length?zs[Math.floor(zs.length/2)]:Math.floor(R.cortes/2);
  corte();tabla();
  if(document.querySelector(".pestanas button.activo").dataset.v==="v3")malla();
  $("#resultado").scrollIntoView({behavior:"smooth"});
}

// cómo se armó el volumen cuando llegó como cortes. Los nombres de archivo los puso el usuario:
// van siempre con textContent, nunca dentro de innerHTML.
function reconstruccion(id){
  const c=R.reconstruccion;$("#rec").classList.toggle("oculto",!c);if(!c)return;
  const n=v=>String(v).replace(".",","),caja=(a,b)=>{const d=document.createElement("div");d.textContent=a;
    const x=document.createElement("b");x.textContent=b;d.append(x);return d;};
  $("#recMeta").replaceChildren(
    caja("Formato",c.formato),caja("Cortes en el volumen",`${c.cortes} (${c.usados} de ${c.recibidos} archivos)`),
    caja("mm por píxel",`${n(c.mm_px[1])} · ${c.fuente_mm_px}`),caja("mm entre cortes",`${n(c.mm_corte)} · ${c.fuente_mm_corte}`));
  const li=(t,b)=>{const e=document.createElement("li");if(b){const x=document.createElement("b");x.textContent=b+" ";e.append(x);}
    e.append(document.createTextNode(t));return e;};
  const det=[li(c.orden,"Orden:"),li(c.sentido||"—","Dónde está la cabeza:")];
  if(c.corte_inferior)det.push(li(`${c.corte_inferior} (pies) … ${c.corte_superior} (cabeza)`,"Extremos:"));
  if(c.conversion)det.push(li(c.conversion,"Intensidad:"));
  if(c.interpolados)det.push(li(`${c.interpolados} cortes interpolados donde faltaban`,"Huecos:"));
  $("#recDetalle").replaceChildren(...det);
  $("#recAvisos").replaceChildren(...(c.avisos||[]).map(a=>li(a)));
  const ign=c.ignorados||[];$("#recIgnorados").classList.toggle("oculto",!ign.length);
  $("#recIgnorados summary").textContent=`${ign.length} archivo(s) o grupos no usados`;
  $("#recIgnorados ul").replaceChildren(...ign.map(a=>li(a)));
  $("#recDescarga").href=`api/trabajo/${id}/volumen`;
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
  $("#gtNota").textContent=R.nota_referencia||"";
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
