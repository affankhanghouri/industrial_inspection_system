const $ = (id) => document.getElementById(id);
const EVENT_LABELS = {HELMET_MISSING:'Helmet Missing',LOADING_AREA_ENTRY:'Loading Area Entry',RESTRICTED_AREA_ENTRY:'Restricted Area Entry'};
const ZONE_LABELS = {loading_area:'Loading Area',restricted_area:'Restricted Area'};
const EXPECTED_CAMERAS = [
  {camera_id:'CAM-001',location:'Refinery Area',state:'OFFLINE'},
  {camera_id:'CAM-002',location:'Loading Area',state:'OFFLINE'},
  {camera_id:'CAM-003',location:'Restricted Area',state:'OFFLINE'}
];
let eventCache=new Map(),cameraCache=new Map(),allEvents=[],selectedStatus='ALL',selectedCamera='ALL';

function esc(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function json(url,options){const response=await fetch(url,options);if(!response.ok)throw Error(await response.text());return response.json();}
function eventName(type){return EVENT_LABELS[type]||String(type||'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());}
function displayName(value){return String(value||'').toLowerCase().replace(/^\w/,c=>c.toUpperCase());}
function cameraLocation(id){return cameraCache.get(id)?.location||EXPECTED_CAMERAS.find(c=>c.camera_id===id)?.location||id;}
function eventLocation(event){return ZONE_LABELS[event.zone_id]||cameraLocation(event.camera_id);}
function localDate(value){return new Intl.DateTimeFormat(undefined,{dateStyle:'medium'}).format(new Date(value));}
function localTime(value){return new Intl.DateTimeFormat(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(value));}

function renderStats(stats){
  const cards=[['Total Violations',stats.total||0],['Open Violations',stats.open||0],['Helmet Violations',stats.HELMET_MISSING||0],['Restricted / Zone Entries',stats.zone_entries||0]];
  $('cards').innerHTML=cards.map(([label,value])=>`<article class="summary-card"><span>${esc(label)}</span><strong>${value}</strong></article>`).join('');
}

function renderCameras(cameras){
  cameraCache=new Map(Object.values(cameras).map(camera=>[camera.camera_id,camera]));
  const display=EXPECTED_CAMERAS.map(expected=>cameraCache.get(expected.camera_id)||expected);
  $('cameras').innerHTML=display.map(camera=>{const online=camera.state==='RUNNING';return `<div class="camera-status ${online?'online':'offline'}"><i></i><span><b>${esc(camera.camera_id)}</b> — ${esc(camera.location||camera.name)}</span><strong>${online?'ONLINE':'OFFLINE'}</strong></div>`;}).join('');
  const value=selectedCamera;
  $('camera-filter').innerHTML='<option value="ALL">All cameras</option>'+display.map(camera=>`<option value="${esc(camera.camera_id)}">${esc(camera.camera_id)} — ${esc(camera.location||camera.name)}</option>`).join('');
  $('camera-filter').value=value;
}

function visibleEvents(){return allEvents.filter(event=>((selectedStatus==='ALL'&&event.status!=='DISMISSED')||event.status===selectedStatus)&&(selectedCamera==='ALL'||event.camera_id===selectedCamera));}
function renderEvents(){
  const events=visibleEvents();eventCache=new Map(allEvents.map(event=>[event.event_id,event]));
  if(!events.length){const filtered=selectedStatus!=='ALL'||selectedCamera!=='ALL';$('events').innerHTML=`<div class="empty-state"><div class="empty-check">✓</div><h3>${filtered?'No violations match these filters.':'No safety violations recorded.'}</h3><p>${filtered?'Try another status or camera.':'All monitored cameras are operating normally.'}</p></div>`;return;}
  $('events').innerHTML=events.map(event=>`<article class="event-card" data-id="${esc(event.event_id)}" tabindex="0" role="button" aria-label="View ${esc(eventName(event.event_type))}">
    <img src="${esc(event.evidence_url)}" alt="Evidence for ${esc(eventName(event.event_type))}" loading="lazy">
    <div class="event-primary"><div class="event-title-row"><h3>${esc(eventName(event.event_type))}</h3><span class="severity severity-${esc(event.severity).toLowerCase()}">${esc(displayName(event.severity))}</span></div><p><b>${esc(event.camera_id)}</b> — ${esc(eventLocation(event))}</p><span class="secondary">Tracking ID: ${esc(event.track_id)}</span></div>
    <div class="event-time"><b>${esc(localTime(event.detected_at))}</b><span>${esc(localDate(event.detected_at))}</span></div>
    <span class="event-status status-${esc(event.status).toLowerCase()}">${esc(displayName(event.status))}</span><span class="view-event">View evidence <b>›</b></span>
    ${event.status!=='DISMISSED'?`<button class="event-dismiss" data-dismiss-id="${esc(event.event_id)}" aria-label="Dismiss ${esc(eventName(event.event_type))}" title="Dismiss this case">×</button>`:''}
  </article>`).join('');
  document.querySelectorAll('.event-card').forEach(card=>{card.onclick=()=>showEvent(card.dataset.id);card.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();showEvent(card.dataset.id);}};});
  document.querySelectorAll('.event-dismiss').forEach(button=>{button.onclick=event=>{event.stopPropagation();dismissFromList(button.dataset.dismissId,button);};});
}

function showEvent(id){
  const event=eventCache.get(id);if(!event)return;$('detail-title').textContent=eventName(event.event_type);
  const video=event.clip_url?`<div class="evidence-video"><h3>Evidence Video</h3><video controls preload="metadata"><source src="${esc(event.clip_url)}" type="video/mp4">Your browser cannot play this evidence video.</video></div>`:`<div class="clip-unavailable"><h3>Evidence Video</h3><p>No video clip is available for this event.</p></div>`;
  $('detail-content').innerHTML=`<div class="detail-meta">
    <div><span>Violation Type</span><b>${esc(eventName(event.event_type))}</b></div><div><span>Severity</span><b class="severity-text-${esc(event.severity).toLowerCase()}">${esc(displayName(event.severity))}</b></div><div><span>Camera</span><b>${esc(event.camera_id)}</b></div>
    <div><span>Location</span><b>${esc(eventLocation(event))}</b></div><div><span>Time</span><b>${esc(localDate(event.detected_at))}, ${esc(localTime(event.detected_at))}</b></div><div><span>Status</span><b>${esc(displayName(event.status))}</b></div><div><span>Tracking ID</span><b>${esc(event.track_id)}</b></div>
  </div><div class="evidence-image"><h3>Screenshot</h3><img src="${esc(event.evidence_url)}" alt="Annotated violation screenshot"></div>${video}${event.status==='OPEN'?`<div class="actions"><button class="resolve" data-action="resolve">Resolve</button><button class="dismiss" data-action="dismiss">Dismiss</button></div>`:''}`;
  $('detail').showModal();document.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>review(id,button.dataset.action,button));
}

async function review(id,action,button){button.disabled=true;try{await json(`/api/events/${id}/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:null})});$('detail').close();await refresh();}catch(error){button.disabled=false;console.error(error);}}
async function dismissFromList(id,button){button.disabled=true;try{await json(`/api/events/${id}/dismiss`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:null})});await refresh();}catch(error){button.disabled=false;console.error(error);}}
async function refresh(){try{const [stats,cameras,events]=await Promise.all([json('/api/stats'),json('/api/cameras'),json('/api/events?limit=100')]);renderStats(stats);renderCameras(cameras);allEvents=events;renderEvents();$('connection').innerHTML='<i></i> System Online';$('connection').className='system-status online';}catch(error){$('connection').innerHTML='<i></i> System Offline';$('connection').className='system-status offline';console.error(error);}}

document.querySelector('.close').onclick=()=>$('detail').close();
$('detail').addEventListener('click',event=>{if(event.target===$('detail'))$('detail').close();});
document.querySelectorAll('[data-status]').forEach(button=>button.onclick=()=>{selectedStatus=button.dataset.status;document.querySelectorAll('[data-status]').forEach(item=>item.classList.toggle('active',item===button));renderEvents();});
$('camera-filter').onchange=event=>{selectedCamera=event.target.value;renderEvents();};
refresh();setInterval(refresh,5000);
