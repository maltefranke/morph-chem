"""Client-side trajectory playback with changing atom counts and bonds."""

import html
import json


def trajectory_viewer(records, sample):
    record = next((r for r in records if r["index"] == int(sample or 0)), None)
    if record is None or not record.get("frames"):
        return "<p style='padding:24px;color:#65758b'>Generate molecules to play their trajectories.</p>"
    frames = json.dumps(record["frames"], separators=(",", ":")).replace("</", "<\\/")
    document = """<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
body{margin:0;font:14px system-ui,sans-serif;color:#26364a;background:#f7f9fc}
#scene{height:clamp(280px,50vw,405px);position:relative;width:100%}
.controls{padding:12px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;border:1px solid #ccd6e3;border-radius:7px;background:white;padding:7px 12px;cursor:pointer;color:#26364a}
#play{background:#3478f6;color:white;border-color:#3478f6;min-width:73px}
#scrub{width:100%;accent-color:#3478f6;margin:0 0 5px}
#readout{font-variant-numeric:tabular-nums;margin-left:auto}
.hint{margin:0;padding:0 14px 12px;color:#697a90;font-size:12px}
</style></head><body>
<div id="scene" aria-label="Molecular trajectory"></div>
<div class="controls">
<input id="scrub" aria-label="Trajectory frame" type="range" min="0" step="1" value="0">
<button id="play">Play</button><button id="restart">Restart</button>
<label>Speed <select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
<label><input type="checkbox" id="loop" checked> Loop</label>
<label><input type="checkbox" id="bonds" checked> Bonds</label>
<span id="readout" aria-live="off"></span>
</div><p class="hint">Drag to rotate · Scroll to zoom · Frames show the sampled states, including atom insertions and deletions.</p>
<script>
const frames=__FRAMES__;
const viewer=$3Dmol.createViewer(document.getElementById('scene'),{backgroundColor:'#f7f9fc',antialias:true});
const scrub=document.getElementById('scrub'),play=document.getElementById('play'),readout=document.getElementById('readout');
scrub.max=frames.length-1;
let frame=0,playing=false,last=0,pausedAtEnd=0;
function draw(index,fit=false){
 frame=Math.max(0,Math.min(frames.length-1,index));
 const camera=fit?null:viewer.getView();
 viewer.removeAllModels();
 // 3Dmol mutates atom objects; each display gets a fresh copy of the raw frame.
 const atoms=frames[frame].atoms.map(a=>({...a,bonds:[...a.bonds],bondOrder:[...a.bondOrder]}));
 viewer.addModel().addAtoms(atoms);
 const style={sphere:{scale:0.23}};
 if(document.getElementById('bonds').checked)style.stick={radius:0.10};
 viewer.setStyle({},style);
 if(fit)viewer.zoomTo();else viewer.setView(camera);
 viewer.render();scrub.value=frame;
 readout.textContent=`t = ${frames[frame].time.toFixed(3)} · N = ${atoms.length}`;
 document.body.dataset.frame=String(frame);document.body.dataset.atomCount=String(atoms.length);
}
function pause(){playing=false;play.textContent='Play';}
play.onclick=()=>{if(playing){pause();return;}if(frame===frames.length-1)draw(0);playing=true;last=0;pausedAtEnd=0;play.textContent='Pause';};
document.getElementById('restart').onclick=()=>{pause();draw(0);};
scrub.oninput=()=>{pause();draw(Number(scrub.value));};
document.getElementById('bonds').onchange=()=>draw(frame);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
function tick(now){
 if(playing){
  const interval=1000/(25*Number(document.getElementById('speed').value));
  if(!last)last=now;
  if(frame===frames.length-1){
   if(!document.getElementById('loop').checked)pause();
   else{if(!pausedAtEnd)pausedAtEnd=now;if(now-pausedAtEnd>=1000){draw(0);last=now;pausedAtEnd=0;}}
  }else if(now-last>=interval){draw(frame+1);last=now;}
 }
 requestAnimationFrame(tick);
}
// Fit to the finished molecule, then keep its camera stable throughout playback.
draw(frames.length-1,true);draw(0);requestAnimationFrame(tick);
window.addEventListener('resize',()=>{viewer.resize();viewer.render();});
</script></body></html>""".replace("__FRAMES__", frames)
    return (
        '<iframe title="Trajectory player" sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:555px;border:0;border-radius:12px" srcdoc="'
        + html.escape(document, quote=True)
        + '"></iframe>'
    )
