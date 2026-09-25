# scratch/update_dashboard_complete.py
"""Carefully and surgically update lung-cancer-xai-phase4/dashboard/index.html
to fully implement Requirements 1-18:
- Connect directly to /api/analyze-ct
- Remove all Math.random(), fake subtypes, fake stages, and hardcoded numbers
- Display dynamic classifications, probabilities, true morphometrics, dynamic Lung-RADS, real XAI heatmaps, true validation metrics, and demo indicator
- Preserve 100% of the dark medical workstation UI styling and layout
"""
from pathlib import Path
import re

dashboard_path = Path(r"c:\Users\ashok\OneDrive\Desktop\Lung Cancer\lung-cancer-xai-phase4\dashboard\index.html")
html = dashboard_path.read_text(encoding="utf-8")

# 1. Update Section 04 and 05 in the right panel to include dynamic Subtype, Staging, and Lung-RADS description
old_sec_04_05 = """  <!-- 04 VERDICT -->
  <div class="verdict-sec">
    <div class="slabel"><span class="num">04 &middot;</span> DIAGNOSTIC <span class="tag">AI Classification</span></div>
    <div class="vmain idle" id="vm">&mdash; &mdash; &mdash;</div>
    <div class="vsubtype" id="vsub">&mdash;</div>
    <div class="vconf-lbl" id="vcl">Run CT analysis to see results</div>

    <div class="crow">
      <div class="clbl">Malignant</div>
      <div class="ctrk"><div class="cfill mg" id="bm"></div></div>
      <div class="cpct" id="pm">&mdash;</div>
    </div>
    <div class="crow">
      <div class="clbl">Benign</div>
      <div class="ctrk"><div class="cfill gn" id="bb"></div></div>
      <div class="cpct" id="pb2">&mdash;</div>
    </div>
    <div class="crow">
      <div class="clbl">Normal</div>
      <div class="ctrk"><div class="cfill cy" id="bn"></div></div>
      <div class="cpct" id="pn">&mdash;</div>
    </div>
    <div class="rbadge" id="rb"></div>
  </div>

  <!-- 05 LUNG-RADS STAGING -->
  <div class="psec" style="padding-bottom:.35rem">
    <div class="slabel"><span class="num">05 &middot;</span> LUNG-RADS <span class="tag">ACR 2022 Staging</span></div>
  </div>
  <div class="stages" id="sstages" style="display:none">
    <div class="stage" id="stS1">Cat 1<br><span style="font-size:.45rem">Neg</span></div>
    <div class="stage" id="stS2">Cat 2<br><span style="font-size:.45rem">Benign</span></div>
    <div class="stage" id="stS3">Cat 3<br><span style="font-size:.45rem">Prob.Ben</span></div>
    <div class="stage" id="stS4">Cat 4A<br><span style="font-size:.45rem">Susp.</span></div>
    <div class="stage" id="stS5">Cat 4B/X<br><span style="font-size:.45rem">Hi Susp</span></div>
  </div>"""

new_sec_04_05 = """  <!-- 04 VERDICT -->
  <div class="verdict-sec">
    <div class="slabel"><span class="num">04 &middot;</span> DIAGNOSTIC <span class="tag">AI Classification</span></div>
    <div class="vmain idle" id="vm">&mdash; &mdash; &mdash;</div>
    <div style="font-size:0.68rem; color:var(--t2); margin-top:2px;" id="vsubRow">Subtype: <span id="vsub">&mdash;</span></div>
    <div style="font-size:0.68rem; color:var(--t2); margin-top:2px;" id="vstageRow">Staging: <span id="vstage">&mdash;</span></div>
    <div class="vconf-lbl" id="vcl">Run CT analysis to see results</div>

    <div class="crow">
      <div class="clbl">Malignant</div>
      <div class="ctrk"><div class="cfill mg" id="bm"></div></div>
      <div class="cpct" id="pm">&mdash;</div>
    </div>
    <div class="crow">
      <div class="clbl">Benign</div>
      <div class="ctrk"><div class="cfill gn" id="bb"></div></div>
      <div class="cpct" id="pb2">&mdash;</div>
    </div>
    <div class="crow">
      <div class="clbl">Normal</div>
      <div class="ctrk"><div class="cfill cy" id="bn"></div></div>
      <div class="cpct" id="pn">&mdash;</div>
    </div>
    <div class="rbadge" id="rb"></div>
  </div>

  <!-- 05 LUNG-RADS STAGING -->
  <div class="psec" style="padding-bottom:.35rem">
    <div class="slabel"><span class="num">05 &middot;</span> LUNG-RADS <span class="tag">ACR 2022 Staging</span></div>
  </div>
  <div class="stages" id="sstages" style="display:none">
    <div class="stage" id="stS1">Cat 1<br><span style="font-size:.45rem">Neg</span></div>
    <div class="stage" id="stS2">Cat 2<br><span style="font-size:.45rem">Benign</span></div>
    <div class="stage" id="stS3">Cat 3<br><span style="font-size:.45rem">Prob.Ben</span></div>
    <div class="stage" id="stS4">Cat 4A<br><span style="font-size:.45rem">Susp.</span></div>
    <div class="stage" id="stS5">Cat 4B/X<br><span style="font-size:.45rem">Hi Susp</span></div>
  </div>
  <div id="radsDesc" style="display:none; padding:0.4rem 0.95rem; font-size:0.63rem; color:var(--t2); line-height:1.4; background:rgba(255,201,71,0.04); border-bottom:1px solid var(--bdr);"></div>"""

assert old_sec_04_05 in html, "Could not find old_sec_04_05"
html = html.replace(old_sec_04_05, new_sec_04_05)

# 2. Add Section 07b Validation and Clinical Disclaimer in the right panel
old_model_sec = """  <!-- 07 MODEL INFO -->
  <div class="psec" style="padding-bottom:.35rem">
    <div class="slabel"><span class="num">07 &middot;</span> MODEL <span class="tag">Architecture</span></div>
  </div>
  <div class="igrid">
    <div class="ichip"><div class="ilbl">Architecture</div><div class="ival" id="iArch">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Parameters</div><div class="ival" id="iPar">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Inference</div><div class="ival" id="iLat">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Input Size</div><div class="ival" id="iSz">&mdash;</div></div>
    <div class="ichip" style="grid-column:1/-1"><div class="ilbl">Checkpoint &amp; Training Set</div><div class="ival" id="iCk">&mdash;</div></div>
  </div>

  <!-- 08 EXPORT -->"""

new_model_sec = """  <!-- 07 MODEL INFO -->
  <div class="psec" style="padding-bottom:.35rem">
    <div class="slabel"><span class="num">07 &middot;</span> MODEL <span class="tag">Architecture</span></div>
  </div>
  <div class="igrid">
    <div class="ichip"><div class="ilbl">Architecture</div><div class="ival" id="iArch">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Parameters</div><div class="ival" id="iPar">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Inference</div><div class="ival" id="iLat">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Input Size</div><div class="ival" id="iSz">&mdash;</div></div>
    <div class="ichip" style="grid-column:1/-1"><div class="ilbl">Checkpoint &amp; Training Set</div><div class="ival" id="iCk">&mdash;</div></div>
  </div>

  <!-- 07b VALIDATION METRICS -->
  <div class="psec" style="padding-bottom:.35rem">
    <div class="slabel"><span class="num">07b &middot;</span> VALIDATION <span class="tag">Empirical Benchmark</span></div>
  </div>
  <div class="igrid" id="valGrid">
    <div class="ichip"><div class="ilbl">AUROC</div><div class="ival" id="valAuroc" style="color:var(--cy)">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Accuracy</div><div class="ival" id="valAcc" style="color:var(--gn)">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Sensitivity</div><div class="ival" id="valSens">&mdash;</div></div>
    <div class="ichip"><div class="ilbl">Specificity</div><div class="ival" id="valSpec">&mdash;</div></div>
    <div class="ichip" style="grid-column:1/-1"><div class="ilbl">Empirical Source</div><div class="ival" id="valSource" style="font-size:0.56rem;color:var(--t3)">&mdash;</div></div>
  </div>

  <!-- 08 EXPORT -->"""

assert old_model_sec in html, "Could not find old_model_sec"
html = html.replace(old_model_sec, new_model_sec)

# 3. Add Clinical Disclaimer Banner at bottom of panel right
old_panel_right_close = """    <button class="ebtn ldct" id="bLdct" onclick="om('mLdct')"   disabled>&#128200; Lung-RADS LDCT Report</button>
  </div>
</aside>"""

new_panel_right_close = """    <button class="ebtn ldct" id="bLdct" onclick="om('mLdct')"   disabled>&#128200; Lung-RADS LDCT Report</button>
  </div>
  <div style="padding:0.75rem 0.95rem; font-size:0.58rem; color:var(--t3); border-top:1px solid var(--bdr); line-height:1.4; background:rgba(0,180,255,0.02);">
    <strong style="color:var(--am);">&#9888; RESEARCH USE ONLY:</strong> AI outputs are assistive predictions and require qualified clinical/radiological interpretation. Never convert an AI prediction into a confirmed diagnosis without histopathology.
  </div>
</aside>"""

assert old_panel_right_close in html, "Could not find old_panel_right_close"
html = html.replace(old_panel_right_close, new_panel_right_close)

# 4. Add Demo Mode badge in Viewer Header
old_vhdr = """  <div class="vhdr">
    <div>
      <div class="slabel" style="margin-bottom:2px"><span class="num">03 &middot;</span> CT VIEWER &mdash; MULTIMODAL XAI</div>
      <div class="vverdict idle" id="vv">Awaiting CT Analysis</div>
      <div class="vverdict-sub" id="vvsub">&mdash;</div>
    </div>
  </div>"""

new_vhdr = """  <div class="vhdr">
    <div style="flex:1">
      <div class="slabel" style="margin-bottom:2px; display:flex; align-items:center; justify-content:space-between;">
        <span><span class="num">03 &middot;</span> CT VIEWER &mdash; MULTIMODAL XAI</span>
        <span class="demo-badge" id="demoBadge" style="display:none; padding:2px 8px; border-radius:12px; background:rgba(255,201,71,0.15); border:1px solid rgba(255,201,71,0.5); color:var(--am); font-size:0.58rem; font-weight:700; letter-spacing:0.05em;">[DEMO MODE: SAMPLE CT]</span>
      </div>
      <div class="vverdict idle" id="vv">Awaiting CT Analysis</div>
      <div class="vverdict-sub" id="vvsub">&mdash;</div>
    </div>
  </div>"""

assert old_vhdr in html, "Could not find old_vhdr"
html = html.replace(old_vhdr, new_vhdr)

# 5. Add morphNotes container under morphometrics
old_mgrid = """    <div class="mcard"><div class="mlbl">Compactness</div><div class="mval" id="mComp">&mdash;</div><div class="msub">sphericity index</div><div class="mbar"><div class="mbfill gn" id="baComp"></div></div></div>
  </div>"""

new_mgrid = """    <div class="mcard"><div class="mlbl">Compactness</div><div class="mval" id="mComp">&mdash;</div><div class="msub">sphericity index</div><div class="mbar"><div class="mbfill gn" id="baComp"></div></div></div>
  </div>
  <div id="morphNotes" style="padding:0.25rem 0.95rem; font-size:0.59rem; color:var(--t3); font-style:italic;"></div>"""

assert old_mgrid in html, "Could not find old_mgrid"
html = html.replace(old_mgrid, new_mgrid)

# 6. Add Explainability note under canvas in viewer
old_vcwrap = """    <!-- DICOM-STYLE CORNER ANNOTATIONS -->
    <div class="vcorner tl" id="ctl" style="display:none"></div>
    <div class="vcorner tr" id="ctr" style="display:none"></div>
    <div class="vcorner bl" id="cbl" style="display:none"></div>
    <div class="vcorner br" id="cbr" style="display:none"></div>
  </div>
  <div class="vfoot">"""

new_vcwrap = """    <!-- DICOM-STYLE CORNER ANNOTATIONS -->
    <div class="vcorner tl" id="ctl" style="display:none"></div>
    <div class="vcorner tr" id="ctr" style="display:none"></div>
    <div class="vcorner bl" id="cbl" style="display:none"></div>
    <div class="vcorner br" id="cbr" style="display:none"></div>
  </div>
  <div style="font-size:0.58rem; color:var(--t3); text-align:center; padding:2px 0;" id="xaiNote">Feature attribution visualizations represent model gradients &amp; attention and are assistive decision-support tools.</div>
  <div class="vfoot">"""

assert old_vcwrap in html, "Could not find old_vcwrap"
html = html.replace(old_vcwrap, new_vcwrap)

# Now update the JavaScript execution logic to be completely dynamic with zero hardcoding!
old_js_section_start = """/* ── RUN PULMONARY AI ANALYSIS ── */
async function run(){"""

# Find start of run()
idx_start = html.find(old_js_section_start)
assert idx_start != -1, "Could not find old_js_section_start"

# Find showRes and addHist and replace up to addHist
old_draw_ov = """function drawOv(tab,op){
  if(!S.done||!S.ov||!S.res)return;
  const r=S.res;
  if(r.label==='normal')return;

  // If we have real XAI heatmaps from neural net, render them onto canvas
  if(S.xaiHeatmaps && (tab==='grad' || tab==='att' || tab==='sal')){
    let hmUrl = null;
    if(tab==='grad') hmUrl = S.xaiHeatmaps.gradcam_pp || S.xaiHeatmaps.gradcam;
    else if(tab==='att') hmUrl = S.xaiHeatmaps.attention;
    else if(tab==='sal') hmUrl = S.xaiHeatmaps.lime || S.xaiHeatmaps.gradcam;

    if(hmUrl){
      const hmImg = new Image();
      hmImg.onload = () => {
        CTX.save();
        CTX.globalAlpha = op;
        CTX.drawImage(hmImg, 0, 0, 512, 512);
        CTX.restore();
      };
      hmImg.src = hmUrl;
      return;
    }
  }

  // Fallback procedural overlay for Segmentation
  CTX.save();CTX.globalAlpha=op;
  const nx=r.label==='malignant'?180:330,ny=220;
  const nr=r.label==='malignant'?24:10;
  CTX.translate(nx,ny);
  const co=r.label==='malignant'?'rgba(255,45,107,':'rgba(0,230,118,';

  if(tab==='seg'){
    CTX.beginPath();
    for(let i=0;i<12;i++){
      const a=(i/12)*Math.PI*2,rad=nr*(1+(Math.random()*.02));
      i===0?CTX.moveTo(rad*Math.cos(a),rad*Math.sin(a)):CTX.lineTo(rad*Math.cos(a),rad*Math.sin(a));
    }
    CTX.closePath();
    const g=CTX.createRadialGradient(0,0,0,0,0,nr*1.2);
    g.addColorStop(0,co+'0.85)');g.addColorStop(.7,co+'0.6)');g.addColorStop(1,co+'0.1)');
    CTX.fillStyle=g;CTX.fill();
    CTX.strokeStyle=r.label==='malignant'?'#ff2d6b':'#00e676';CTX.lineWidth=2;CTX.stroke();
    // Crosshair measurement
    const ds=nr*1.8;
    CTX.strokeStyle=r.label==='malignant'?'#ff2d6b':'#00e676';CTX.lineWidth=1;CTX.setLineDash([3,3]);
    CTX.beginPath();CTX.moveTo(-ds,0);CTX.lineTo(ds,0);CTX.stroke();
    CTX.beginPath();CTX.moveTo(0,-ds);CTX.lineTo(0,ds);CTX.stroke();
    CTX.setLineDash([]);
  }
  CTX.restore();
}"""

new_draw_ov = """function drawOv(tab, op){
  if(!S.done || !S.ov || !S.res) return;
  
  let overlayUrl = null;
  if(tab === 'seg' && S.xaiHeatmaps && S.xaiHeatmaps.segmentation){
    overlayUrl = S.xaiHeatmaps.segmentation;
  } else if(tab === 'grad' && S.xaiHeatmaps){
    overlayUrl = S.xaiHeatmaps.gradcam_pp || S.xaiHeatmaps.gradcam;
  } else if(tab === 'att' && S.xaiHeatmaps){
    overlayUrl = S.xaiHeatmaps.attention;
  } else if(tab === 'sal' && S.xaiHeatmaps){
    overlayUrl = S.xaiHeatmaps.saliency || S.xaiHeatmaps.lime;
  }

  if(overlayUrl){
    const hmImg = new Image();
    hmImg.onload = () => {
      CTX.save();
      CTX.globalAlpha = op;
      CTX.drawImage(hmImg, 0, 0, 512, 512);
      CTX.restore();
    };
    hmImg.src = overlayUrl;
  }
}"""

assert old_draw_ov in html, "Could not find old_draw_ov"
html = html.replace(old_draw_ov, new_draw_ov)

# Now replace run() and showRes()
old_run_and_showres_regex = r"/\* ── RUN PULMONARY AI ANALYSIS ── \*/[\s\S]+?function addHist\(r\)\{"

new_run_and_showres = """/* ── RUN PULMONARY AI ANALYSIS (DYNAMIC RESEARCH PIPELINE) ── */
async function run(){
  if(!S.loaded || S.busy) return;
  S.busy = true; S.done = false;
  ss('proc', 'Analysing CT with Hybrid CNN-ViT…');
  document.getElementById('btnA').disabled = true;
  document.getElementById('idle').classList.add('gone');
  document.getElementById('pov').classList.add('on');

  const ps = ['ps1','ps2','ps3','ps4'], pcts = [20, 50, 80, 95];
  for(let i=0; i<4; i++){
    ps.forEach((p,j)=>{
      const el = document.getElementById(p);
      el.style.display = 'block';
      el.className = 'pstep' + (j===i ? ' cur' : '');
    });
    document.getElementById('pb').style.width = pcts[i] + '%';
    await sl(300);
  }

  let data = null;
  try {
    const form = new FormData();
    if(S.activeFile){
      form.append('file', S.activeFile);
    } else if(S.activeSampleId){
      form.append('sample_id', S.activeSampleId);
    } else {
      form.append('sample_id', 'malignant');
    }
    form.append('architecture', 'hybrid');

    const res = await fetch(`${API_BASE}/api/analyze-ct`, {
      method: 'POST',
      body: form
    });

    if(!res.ok){
      const errTxt = await res.text();
      throw new Error(`Inference server returned ${res.status}: ${errTxt}`);
    }
    data = await res.json();
  } catch(err) {
    console.error('Inference error:', err);
    document.getElementById('pov').classList.remove('on');
    S.busy = false;
    document.getElementById('btnA').disabled = false;
    ss('err', 'Inference Service Unavailable');
    toast(`Inference failed: ${err.message}. Never fabricating medical results.`, 'e', 6000);
    document.getElementById('vv').className = 'vverdict err';
    document.getElementById('vv').textContent = 'ANALYSIS UNAVAILABLE';
    document.getElementById('vvsub').textContent = 'Model inference failed. No results fabricated.';
    document.getElementById('vm').className = 'vmain err';
    document.getElementById('vm').textContent = 'ERROR';
    document.getElementById('vcl').textContent = 'FastAPI server unreachable or inference exception.';
    return;
  }

  document.getElementById('pb').style.width = '100%';
  await sl(200);
  document.getElementById('pov').classList.remove('on');

  // Save actual model heatmaps & segmentation
  S.xaiHeatmaps = {
    segmentation: data.segmentation ? data.segmentation.mask_base64 : null,
    gradcam: data.explainability ? data.explainability.gradcam_base64 : null,
    gradcam_pp: data.explainability ? data.explainability.gradcam_pp_base64 : null,
    attention: data.explainability ? data.explainability.attention_base64 : null,
    saliency: data.explainability ? data.explainability.saliency_base64 : null
  };

  S.res = data;
  S.done = true;
  S.busy = false;
  drawBase();
  setTimeout(()=>drawOv(S.tab, S.op), 80);
  showRes(data);
  addHist(data);
  ss('ready', 'Radiology AI Ready');
  toast(`Analysis complete: ${data.classification.predicted_class.toUpperCase()} (${data.lungRads.category})`, 's');
}

function sl(ms){ return new Promise(r=>setTimeout(r, ms)); }

function showRes(data){
  const c = data.classification;
  const pred = c.predicted_class;
  const lbl = pred.toLowerCase();
  const confPct = (c.confidence * 100).toFixed(1);

  // Demo badge indicator
  const isDemo = data.is_demo || (data.metadata && data.metadata.is_demo_mode);
  const demoBadge = document.getElementById('demoBadge');
  if(demoBadge){
    demoBadge.style.display = isDemo ? 'inline-block' : 'none';
  }

  // Viewer Header
  const vv = document.getElementById('vv');
  vv.className = 'vverdict ' + lbl;
  vv.textContent = `${pred.toUpperCase()} · ${confPct}% (AI Prediction)`;

  const vvsub = document.getElementById('vvsub');
  const subtext = data.subtyping.available ? data.subtyping.subtype : 'Subtype: Not evaluated by 3-class model';
  vvsub.textContent = `${data.lungRads.category} · ${subtext}`;

  // Right Panel Diagnostic Verdict
  const vm = document.getElementById('vm');
  vm.className = 'vmain ' + lbl;
  vm.textContent = pred.toUpperCase();

  // Subtype (Dynamic - Never fabricated)
  const vsub = document.getElementById('vsub');
  if(data.subtyping.available && data.subtyping.subtype){
    vsub.textContent = data.subtyping.subtype;
  } else {
    vsub.innerHTML = `<span style="color:var(--t3);font-style:italic;">${data.subtyping.reason || 'Not evaluated by current 3-class model'}</span>`;
  }

  // Staging (Dynamic - Never fabricated)
  const vstage = document.getElementById('vstage');
  if(vstage){
    if(data.staging.available && data.staging.stage){
      vstage.textContent = data.staging.stage;
    } else {
      vstage.innerHTML = `<span style="color:var(--t3);font-style:italic;">${data.staging.reason || 'Not determined from 2D slice'}</span>`;
    }
  }

  // Confidence & probabilities
  document.getElementById('vcl').textContent = `${confPct}% model confidence · ${data.model_info.architecture}`;

  const probs = c.probabilities;
  const pNorm = (probs.Normal || probs.normal || 0) * 100;
  const pBen = (probs.Benign || probs.benign || 0) * 100;
  const pMal = (probs.Malignant || probs.malignant || 0) * 100;

  setTimeout(()=>{
    document.getElementById('bn').style.width = pNorm + '%';
    document.getElementById('bb').style.width = pBen + '%';
    document.getElementById('bm').style.width = pMal + '%';
    document.getElementById('pn').textContent = pNorm.toFixed(1) + '%';
    document.getElementById('pb2').textContent = pBen.toFixed(1) + '%';
    document.getElementById('pm').textContent = pMal.toFixed(1) + '%';
  }, 100);

  // Clinical risk indication badge
  const rb = document.getElementById('rb');
  rb.style.display = 'block';
  if(lbl === 'malignant'){
    rb.className = 'rbadge hi';
    rb.textContent = '⚠ AI PREDICTION: MALIGNANT — URGENT CLINICAL / RADIOLOGICAL REVIEW';
  } else if(lbl === 'benign'){
    rb.className = 'rbadge mod';
    rb.textContent = '◈ AI PREDICTION: BENIGN — RADIOLOGICAL CORRELATION ADVISED';
  } else {
    rb.className = 'rbadge lo';
    rb.textContent = '✓ AI PREDICTION: NORMAL — REGULAR SURVEILLANCE';
  }

  // Dynamic Lung-RADS Staging
  document.getElementById('sstages').style.display = 'flex';
  const stageIdx = data.lungRads.stage_idx;
  ['stS1','stS2','stS3','stS4','stS5'].forEach((id, i)=>{
    document.getElementById(id).classList.toggle('active', i === stageIdx);
  });
  const radsDesc = document.getElementById('radsDesc');
  if(radsDesc){
    radsDesc.style.display = 'block';
    radsDesc.innerHTML = `<strong style="color:var(--am);">${data.lungRads.category} (${data.lungRads.assessment}):</strong> ${data.lungRads.clinical_criteria} <em>${data.lungRads.recommendation}</em>`;
  }

  // Dynamic Morphometrics
  const m = data.measurements;
  document.getElementById('mDiam').textContent = m.max_diameter_mm !== null ? `${m.max_diameter_mm}` : '—';
  document.getElementById('mVol').textContent = m.volume_mm3 !== null ? `${m.volume_mm3.toLocaleString()}` : '—';
  document.getElementById('mHU').textContent = m.mean_hu !== null ? `${m.mean_hu}` : '—';
  document.getElementById('mSpic').textContent = typeof m.spiculation_index === 'number' ? m.spiculation_index.toFixed(2) : '0.00';
  document.getElementById('mGGO').textContent = typeof m.ggo_ratio_pct === 'number' ? `${m.ggo_ratio_pct}%` : '—';
  document.getElementById('mComp').textContent = typeof m.compactness_index === 'number' ? m.compactness_index.toFixed(2) : '0.00';

  const morphNotes = document.getElementById('morphNotes');
  if(morphNotes && m.notes){
    morphNotes.textContent = m.notes;
  }

  setTimeout(()=>{
    const diamBar = m.max_diameter_mm !== null ? Math.min((m.max_diameter_mm / 40) * 100, 100) : 0;
    const volBar = m.volume_mm3 !== null ? Math.min((m.volume_mm3 / 15000) * 100, 100) : 0;
    const huNorm = m.mean_hu !== null ? Math.min(Math.max((m.mean_hu + 1000) / 2000 * 100, 0), 100) : 0;
    document.getElementById('baDiam').style.width = diamBar + '%';
    document.getElementById('baVol').style.width = volBar + '%';
    document.getElementById('baHU').style.width = huNorm + '%';
    document.getElementById('baSpic').style.width = ((m.spiculation_index || 0) * 100) + '%';
    document.getElementById('baGGO').style.width = (m.ggo_ratio_pct || 0) + '%';
    document.getElementById('baComp').style.width = ((m.compactness_index || 0) * 100) + '%';
  }, 300);

  // Dynamic Lobe Involvement Map
  const lobeMap = data.lobes || {};
  const lobeKeys = {RUL:'lRUL', RML:'lRML', RLL:'lRLL', LUL:'lLUL', Ling:'lLing', LLL:'lLLL'};
  const lobeChips = document.getElementById('lobeGrid').querySelectorAll('.lobe-chip');
  Object.keys(lobeKeys).forEach((k, idx)=>{
    const el = document.getElementById(lobeKeys[k]);
    const val = lobeMap[k] || 'Clear';
    el.textContent = val;
    const chip = lobeChips[idx];
    chip.className = 'lobe-chip';
    if(val === 'Clear') chip.classList.add('clear');
    else if(val === 'Primary Lesion' || val === 'Primary') chip.classList.add('affected');
    else if(val.includes('Involved') || val.includes('Malignant')) chip.classList.add('affected');
    else chip.classList.add('clear');
  });

  // Dynamic CT Metadata in Left Panel
  const meta = data.metadata || {};
  document.getElementById('metaMod').textContent = meta.modality || 'Thoracic CT';
  document.getElementById('metaSlice').textContent = meta.slice_thickness || 'Not available';
  document.getElementById('metaKvp').textContent = meta.kvp || 'Not available';
  document.getElementById('metaWL').textContent = meta.window_level || 'Standard Thoracic';
  document.getElementById('metaRads').textContent = data.lungRads.code || data.lungRads.category;

  // Dynamic Model Information
  const mi = data.model_info || {};
  document.getElementById('iArch').textContent = mi.architecture || 'Hybrid CNN-ViT';
  document.getElementById('iPar').textContent = mi.parameters_formatted || '—';
  document.getElementById('iLat').textContent = (mi.latency_ms || 0).toFixed(1) + ' ms';
  document.getElementById('iSz').textContent = mi.input_size || '224×224×3';
  document.getElementById('iCk').textContent = `${mi.checkpoint} (${mi.training_set || 'Trained'})`;

  // Dynamic Empirical Validation Metrics
  const val = data.validation || {};
  const valAuroc = document.getElementById('valAuroc');
  if(valAuroc && val.roc_auc !== undefined){
    valAuroc.textContent = (val.roc_auc * 100).toFixed(1) + '%';
    document.getElementById('valAcc').textContent = (val.accuracy * 100).toFixed(1) + '%';
    document.getElementById('valSens').textContent = (val.sensitivity * 100).toFixed(1) + '%';
    document.getElementById('valSpec').textContent = (val.specificity * 100).toFixed(1) + '%';
    document.getElementById('valSource').textContent = val.source || 'Empirical Validation';
  }

  // Dynamic Footer Info
  document.getElementById('flatency').textContent = `Inference ${mi.latency_ms || 0} ms (${mi.device || 'CPU'})`;
  document.getElementById('fseed').textContent = meta.source_type || 'Study Processed';

  // Dynamic DICOM Corner Annotations
  ['ctl','ctr','cbl','cbr'].forEach(id=>{ document.getElementById(id).style.display = 'block'; });
  document.getElementById('ctl').textContent = `LungXAI Research AI\\n${mi.architecture || 'Hybrid CNN-ViT'}`;
  document.getElementById('ctr').textContent = `${meta.modality}\\n${meta.slice_thickness}`;
  document.getElementById('cbl').textContent = `${meta.window_level}\\n${meta.patient_id || 'ANONYMOUS'}`;
  document.getElementById('cbr').textContent = `${data.lungRads.code}\\n${pred} ${confPct}%`;
  document.getElementById('ruler').classList.add('vis');

  document.getElementById('bRep').disabled = false;
  document.getElementById('bDcm').disabled = false;
  document.getElementById('bLdct').disabled = false;
}

function addHist(r){"""

html, n_subs = re.subn(old_run_and_showres_regex, new_run_and_showres, html, count=1)
assert n_subs == 1, f"Regex substitution failed: {n_subs}"

# 7. Update addHist, renderHist, and exports to handle data schema properly
old_hist_code = """function addHist(r){
  S.hist.unshift({...r,time:new Date().toLocaleTimeString('en-GB',{hour12:false}),id:Date.now()});
  renderHist();
}

function renderHist(){
  const list=document.getElementById('hlist');
  document.getElementById('hempty').style.display=S.hist.length?'none':'block';
  list.querySelectorAll('.hcard').forEach(e=>e.remove());
  S.hist.forEach(e=>{
    const d=document.createElement('div');d.className='hcard '+e.label;
    d.onclick=()=>replayHist(e);
    const conf=(e.probs[e.label]*100).toFixed(1);
    d.innerHTML='<div class="hthumb"><canvas id="ht'+e.id+'" width="34" height="34"></canvas></div>'
      +'<div class="hmeta"><div class="hverd">'+e.label.toUpperCase()+'</div>'
      +'<div class="htime">'+e.time+' &middot; '+e.rads+'</div>'
      +'<div class="hsubtype" style="font-size:.52rem;color:var(--t3)">'+e.seed+'</div></div>'
      +'<div class="hconf">'+conf+'%</div>';
    list.appendChild(d);
  });
}

function replayHist(e){
  S.done=true;S.res=e;drawBase();setTimeout(()=>drawOv(S.tab,S.op),80);showRes(e);
  toast('Replaying study: '+e.rads+' '+e.label.toUpperCase(),'i');
}"""

new_hist_code = """function addHist(r){
  const label = r.classification ? r.classification.predicted_class.toLowerCase() : (r.label || 'unknown');
  const conf = r.classification ? (r.classification.confidence * 100).toFixed(1) : '—';
  const rads = r.lungRads ? r.lungRads.category : (r.rads || '—');
  const seed = r.metadata ? (r.metadata.source_type || r.metadata.patient_id) : (r.seed || '—');
  
  S.hist.unshift({
    data: r,
    label: label,
    conf: conf,
    rads: rads,
    seed: seed,
    time: new Date().toLocaleTimeString('en-GB', {hour12:false}),
    id: Date.now()
  });
  renderHist();
}

function renderHist(){
  const list = document.getElementById('hlist');
  document.getElementById('hempty').style.display = S.hist.length ? 'none' : 'block';
  list.querySelectorAll('.hcard').forEach(e=>e.remove());
  S.hist.forEach(e=>{
    const d = document.createElement('div');
    d.className = 'hcard ' + e.label;
    d.onclick = ()=>replayHist(e.data);
    d.innerHTML = '<div class="hthumb"><canvas id="ht'+e.id+'" width="34" height="34"></canvas></div>'
      + '<div class="hmeta"><div class="hverd">' + e.label.toUpperCase() + '</div>'
      + '<div class="htime">' + e.time + ' &middot; ' + e.rads + '</div>'
      + '<div class="hsubtype" style="font-size:.52rem;color:var(--t3)">' + e.seed + '</div></div>'
      + '<div class="hconf">' + e.conf + '%</div>';
    list.appendChild(d);
  });
}

function replayHist(data){
  S.done = true; S.res = data;
  S.xaiHeatmaps = {
    segmentation: data.segmentation ? data.segmentation.mask_base64 : null,
    gradcam: data.explainability ? data.explainability.gradcam_base64 : null,
    gradcam_pp: data.explainability ? data.explainability.gradcam_pp_base64 : null,
    attention: data.explainability ? data.explainability.attention_base64 : null,
    saliency: data.explainability ? data.explainability.saliency_base64 : null
  };
  drawBase();
  setTimeout(()=>drawOv(S.tab, S.op), 80);
  showRes(data);
  toast('Replaying study: ' + (data.lungRads ? data.lungRads.category : '') + ' ' + (data.classification ? data.classification.predicted_class.toUpperCase() : ''), 'i');
}"""

assert old_hist_code in html, "Could not find old_hist_code"
html = html.replace(old_hist_code, new_hist_code)

# 8. Update Export functions to dynamically use data fields
old_export_funcs = """/* ── EXPORTS ── */
function doExport(){
  cm('mReport');
  if(!S.res){toast('Please run analysis first','w');return}
  const r=S.res;
  const repHtml=`<!DOCTYPE html><html><head><meta charset="utf-8"><title>LungXAI Clinical Report - ${r.rads}</title>
  <style>body{font-family:sans-serif;background:#060d1a;color:#e8f4ff;padding:2rem;}
  .box{max-width:800px;margin:0 auto;background:#09121f;border:1px solid #00c8ff;border-radius:12px;padding:2rem;}
  h1{color:#00c8ff;margin-top:0;}
  table{width:100%;border-collapse:collapse;margin-top:1rem;}
  th,td{border:1px solid rgba(0,200,255,.2);padding:.6rem;text-align:left;}
  th{background:rgba(0,200,255,.1);color:#00c8ff;}
  .badge{padding:.3rem .8rem;border-radius:4px;font-weight:bold;text-transform:uppercase;}
  .badge.malignant{background:#ff2d6b;color:#fff;}
  .badge.benign{background:#00e676;color:#000;}
  .badge.normal{background:#00c8ff;color:#000;}
  </style></head><body><div class="box">
  <h1>🫁 LungXAI Pulmonary Oncology Report</h1>
  <p><strong>Series:</strong> ${r.seed} | <strong>Model:</strong> ${r.arch} | <strong>Protocol:</strong> ${r.meta.kvp} (${r.meta.slice})</p>
  <div style="margin:1rem 0;"><span class="badge ${r.label}">${r.label} &middot; ${r.rads}</span></div>
  <p><strong>Diagnosis:</strong> ${r.subtype}</p>
  <h3>Ternary Probabilities:</h3>
  <p>Malignant: ${(r.probs.malignant*100).toFixed(1)}% | Benign: ${(r.probs.benign*100).toFixed(1)}% | Normal: ${(r.probs.normal*100).toFixed(1)}%</p>
  <h3>Nodule Morphometrics:</h3>
  <table>
    <tr><th>Max Diameter</th><td>${r.morph.diam} mm</td><th>Volume</th><td>${r.morph.vol} mm³</td></tr>
    <tr><th>Mean HU</th><td>${r.morph.hu} HU</td><th>Spiculation</th><td>${r.morph.spic}</td></tr>
    <tr><th>GGO Ratio</th><td>${r.morph.ggo}%</td><th>Compactness</th><td>${r.morph.compact}</td></tr>
  </table>
  <h3>Lung Lobe Involvement:</h3>
  <p>RUL: ${r.lobes.RUL} | RML: ${r.lobes.RML} | RLL: ${r.lobes.RLL} | LUL: ${r.lobes.LUL} | Lingula: ${r.lobes.Ling} | LLL: ${r.lobes.LLL}</p>
  <hr style="border:none;border-top:1px solid rgba(0,200,255,.2);margin:1.5rem 0;">
  <p style="font-size:.75rem;color:#7db8d4;">ACR Lung-RADS 2022 Certified AI Summary &middot; Generated by LungXAI Hybrid CNN-ViT Engine</p>
  </div></body></html>`;

  const blob=new Blob([repHtml],{type:'text/html'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;a.download='LungXAI_Report_'+r.label.toUpperCase()+'_'+Date.now()+'.html';
  document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
  toast('Pulmonary AI Report downloaded!','s');
}

function doDicom(){
  cm('mDicom');
  if(!S.res){toast('Please run analysis first','w');return}
  const r=S.res;
  const dcmPayload={
    "DICOM_SR_TID": 1500,
    "PatientID": "ANONYMOUS_LUNG_01",
    "StudyInstanceUID": "1.2.840.10008.5.1.4.1.1."+Date.now(),
    "Modality": r.meta.mod,
    "AlgorithmName": "LungXAI_Hybrid_CNN_ViT",
    "Findings": {
      "Category": r.label.toUpperCase(),
      "LungRADS": r.rads,
      "ProbabilityMalignancy": r.probs.malignant,
      "Morphometrics": r.morph,
      "LobeInvolvement": r.lobes
    },
    "Timestamp": new Date().toISOString()
  };
  const blob=new Blob([JSON.stringify(dcmPayload,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;a.download='LungXAI_DICOM_SR_TID1500_'+Date.now()+'.json';
  document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
  toast('DICOM Structured Report exported!','s');
}

function doLdct(){
  cm('mLdct');
  if(!S.res){toast('Please run analysis first','w');return}
  const r=S.res;
  const ldctHtml=`<!DOCTYPE html><html><head><meta charset="utf-8"><title>ACR Lung-RADS Screening Report</title>
  <style>body{font-family:sans-serif;background:#060d1a;color:#e8f4ff;padding:2rem;}
  .box{max-width:800px;margin:0 auto;background:#09121f;border:1px solid #ffc947;border-radius:12px;padding:2rem;}
  h1{color:#ffc947;margin-top:0;}
  .rec{background:rgba(255,201,71,.1);border:1px solid #ffc947;padding:1rem;border-radius:6px;margin:1rem 0;}
  </style></head><body><div class="box">
  <h1>ACR Lung-RADS v2022 Screening Summary</h1>
  <p><strong>Screening Assessment:</strong> ${r.rads}</p>
  <p><strong>Primary AI Classification:</strong> ${r.label.toUpperCase()}</p>
  <div class="rec">
    <strong>Clinical Management Recommendation:</strong><br>
    ${r.label==='malignant'?'High suspicion for lung cancer. Urgent multidisciplinary thoracic oncology consultation, PET-CT, and tissue biopsy recommended.':r.label==='benign'?'Benign appearance. 6-month low-dose chest CT follow-up to assess stability.':'Negative scan. Routine annual low-dose screening examination recommended.'}
  </div>
  <p>Nodule Max Axial Diameter: ${r.morph.diam} mm</p>
  <p>Fleischner Society Risk Tier: ${r.label==='malignant'?'High Risk (>8mm spiculated)':'Low/Moderate Risk'}</p>
  </div></body></html>`;

  const blob=new Blob([ldctHtml],{type:'text/html'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;a.download='LungRADS_ACR2022_'+r.rads+'_'+Date.now()+'.html';
  document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
  toast('Lung-RADS LDCT Report downloaded!','s');
}"""

new_export_funcs = """/* ── EXPORTS (DYNAMIC FROM RESEARCH PIPELINE) ── */
function doExport(){
  cm('mReport');
  if(!S.res){toast('Please run analysis first','w');return}
  const r = S.res;
  const pred = r.classification.predicted_class;
  const lbl = pred.toLowerCase();
  const rads = r.lungRads.category;
  const m = r.measurements;
  const mi = r.model_info;
  const meta = r.metadata;
  const probs = r.classification.probabilities;

  const repHtml = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>LungXAI Clinical Report - ${rads}</title>
  <style>body{font-family:sans-serif;background:#060d1a;color:#e8f4ff;padding:2rem;}
  .box{max-width:800px;margin:0 auto;background:#09121f;border:1px solid #00c8ff;border-radius:12px;padding:2rem;}
  h1{color:#00c8ff;margin-top:0;}
  table{width:100%;border-collapse:collapse;margin-top:1rem;}
  th,td{border:1px solid rgba(0,200,255,.2);padding:.6rem;text-align:left;}
  th{background:rgba(0,200,255,.1);color:#00c8ff;}
  .badge{padding:.3rem .8rem;border-radius:4px;font-weight:bold;text-transform:uppercase;}
  .badge.malignant{background:#ff2d6b;color:#fff;}
  .badge.benign{background:#00e676;color:#000;}
  .badge.normal{background:#00c8ff;color:#000;}
  .disclaimer{font-size:0.75rem;color:#7db8d4;margin-top:1.5rem;padding:0.8rem;background:rgba(0,180,255,0.05);border-left:3px solid #ffc947;}
  </style></head><body><div class="box">
  <h1>🫁 LungXAI AI-Assisted Pulmonary CT Report</h1>
  <p><strong>Patient/Study:</strong> ${meta.patient_id} | <strong>Protocol:</strong> ${meta.modality} (${meta.slice_thickness}) | <strong>Model:</strong> ${mi.architecture}</p>
  <div style="margin:1rem 0;"><span class="badge ${lbl}">AI PREDICTION: ${pred} &middot; ${rads}</span></div>
  <p><strong>Subtype Status:</strong> ${r.subtyping.available ? r.subtyping.subtype : r.subtyping.reason}</p>
  <p><strong>Clinical Staging:</strong> ${r.staging.available ? r.staging.stage : r.staging.reason}</p>
  <h3>Ternary AI Classification Probabilities:</h3>
  <p>Malignant: ${((probs.Malignant||probs.malignant||0)*100).toFixed(1)}% | Benign: ${((probs.Benign||probs.benign||0)*100).toFixed(1)}% | Normal: ${((probs.Normal||probs.normal||0)*100).toFixed(1)}%</p>
  <h3>Quantitative Nodule Morphometrics:</h3>
  <table>
    <tr><th>Max Diameter</th><td>${m.max_diameter_mm !== null ? m.max_diameter_mm + ' mm' : 'None detected'}</td><th>Volume</th><td>${m.volume_mm3 !== null ? m.volume_mm3 + ' mm³' : '—'}</td></tr>
    <tr><th>Mean HU</th><td>${m.mean_hu !== null ? m.mean_hu + ' HU' : '—'}</td><th>Spiculation</th><td>${m.spiculation_index}</td></tr>
    <tr><th>GGO Ratio</th><td>${m.ggo_ratio_pct}%</td><th>Compactness</th><td>${m.compactness_index}</td></tr>
  </table>
  <h3>Anatomical Lobe Mapping:</h3>
  <p>RUL: ${r.lobes.RUL} | RML: ${r.lobes.RML} | RLL: ${r.lobes.RLL} | LUL: ${r.lobes.LUL} | Lingula: ${r.lobes.Lingula||r.lobes.Ling} | LLL: ${r.lobes.LLL}</p>
  <div class="disclaimer">
    <strong>DISCLAIMER:</strong> RESEARCH USE ONLY. This AI-assisted prediction is an investigative decision-support tool and requires qualified clinical/radiological interpretation.
  </div>
  </div></body></html>`;

  const blob = new Blob([repHtml], {type:'text/html'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `LungXAI_Report_${pred}_${Date.now()}.html`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  toast('Pulmonary AI Report downloaded!', 's');
}

function doDicom(){
  cm('mDicom');
  if(!S.res){toast('Please run analysis first','w');return}
  const r = S.res;
  const dcmPayload = {
    "DICOM_SR_TID": 1500,
    "PatientID": r.metadata ? r.metadata.patient_id : "ANONYMOUS_STUDY",
    "StudyInstanceUID": "1.2.840.10008.5.1.4.1.1." + Date.now(),
    "Modality": r.metadata ? r.metadata.modality : "CT",
    "AlgorithmName": r.model_info ? r.model_info.architecture : "LungXAI_Hybrid_CNN_ViT",
    "Findings": {
      "Category": r.classification.predicted_class,
      "Confidence": r.classification.confidence,
      "Probabilities": r.classification.probabilities,
      "Subtyping": r.subtyping,
      "Staging": r.staging,
      "LungRADS": r.lungRads,
      "Morphometrics": r.measurements,
      "LobeInvolvement": r.lobes,
      "Validation": r.validation
    },
    "Disclaimer": r.clinical_disclaimer,
    "Timestamp": new Date().toISOString()
  };
  const blob = new Blob([JSON.stringify(dcmPayload, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `LungXAI_DICOM_SR_TID1500_${Date.now()}.json`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  toast('DICOM Structured Report exported!', 's');
}

function doLdct(){
  cm('mLdct');
  if(!S.res){toast('Please run analysis first','w');return}
  const r = S.res;
  const rads = r.lungRads;
  const pred = r.classification.predicted_class;
  const m = r.measurements;

  const ldctHtml = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>ACR Lung-RADS Screening Report</title>
  <style>body{font-family:sans-serif;background:#060d1a;color:#e8f4ff;padding:2rem;}
  .box{max-width:800px;margin:0 auto;background:#09121f;border:1px solid #ffc947;border-radius:12px;padding:2rem;}
  h1{color:#ffc947;margin-top:0;}
  .rec{background:rgba(255,201,71,.1);border:1px solid #ffc947;padding:1rem;border-radius:6px;margin:1rem 0;}
  </style></head><body><div class="box">
  <h1>ACR Lung-RADS v2022 Screening Summary</h1>
  <p><strong>Screening Assessment:</strong> ${rads.category} (${rads.assessment})</p>
  <p><strong>Primary AI Assistive Classification:</strong> ${pred}</p>
  <div class="rec">
    <strong>Clinical Management Criteria & Recommendation:</strong><br>
    ${rads.clinical_criteria} &mdash; ${rads.recommendation}
  </div>
  <p><strong>Nodule Max Diameter:</strong> ${m.max_diameter_mm !== null ? m.max_diameter_mm + ' mm' : 'No focal nodule measured'}</p>
  <p><strong>Mean Lesion HU:</strong> ${m.mean_hu !== null ? m.mean_hu + ' HU' : '—'}</p>
  <p style="font-size:0.75rem; color:#7db8d4; margin-top:1.5rem;">${r.clinical_disclaimer}</p>
  </div></body></html>`;

  const blob = new Blob([ldctHtml], {type:'text/html'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `LungRADS_ACR2022_${rads.code || 'LR'}_${Date.now()}.html`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  toast('Lung-RADS LDCT Report downloaded!', 's');
}"""

assert old_export_funcs in html, "Could not find old_export_funcs"
html = html.replace(old_export_funcs, new_export_funcs)

dashboard_path.write_text(html, encoding="utf-8")
print("SUCCESS: Updated dashboard/index.html cleanly!")
