'use strict';
/* 漂流助手控制台。界面按 ALAS 的组织方式:侧栏任务分组 + 概览页的调度器 / 日志 / 任务组。
   API 仍是本机 HTTP 接口,patch 与测试依赖的端点、token、页面标题都不变。 */
const $ = id => document.getElementById(id);
const paths = {
  sail: '<path d="M12 3v14H3L12 3Zm3 4 6 10h-6M3 20c2 2 4-2 6 0s4-2 6 0 4-2 6 0"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  parachute: '<path d="M3 11a9 9 0 0 1 18 0c-2-2-4-2-6 0-2-2-4-2-6 0-2-2-4-2-6 0ZM3 11l7 8m11-8-7 8M9 11l2 8m4-8-2 8M12 3c-3 2-4 5-3 8m3-8c3 2 4 5 3 8"/><rect x="10" y="19" width="4" height="3" rx="1"/>',
  potion: '<path d="M9 3h6M10 3v5l-5 8a4 4 0 0 0 3 5h8a4 4 0 0 0 3-5l-5-8V3M7 14h10"/><path d="m12 11-2 5h4l-2 3"/>',
  crown: '<path d="m3 6 4 5 5-7 5 7 4-5-2 13H5L3 6Zm2 10h14"/><circle cx="3" cy="5" r="1"/><circle cx="12" cy="3" r="1"/><circle cx="21" cy="5" r="1"/>',
  mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 6 9 7 9-7M3 18l6-6m12 6-6-6"/>',
  gift: '<rect x="3" y="8" width="18" height="4" rx="1"/><path d="M5 12v9h14v-9M12 8v13M12 8H8a3 3 0 1 1 3-3l1 3Zm0 0h4a3 3 0 1 0-3-3l-1 3Z"/>',
  coin: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="6"/><path d="M14 7a5 5 0 1 0 3 7 5 5 0 0 1-3-7Z"/>',
  swords: '<path d="m4 3 3 1 12 13-2 2L4 7 4 3Zm16 0-3 1-5 5m-3 3-4 5 2 2 4-4M3 17l4 4m10 0 4-4M3 21l2-2m14 0 2 2"/>',
  portal: '<circle cx="12" cy="12" r="9"/><path d="M17 7c-3-2-8-1-9 3-1 3 1 6 4 6 2 0 4-2 3-4-1-2-4-1-3 1m-7 5 2-2m10-9 2-2"/>',
  growth: '<path d="m12 3-9 9h5v9h8v-9h5L12 3Z"/>',
  flask: '<path d="M9 3h6v4l4 6v6a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-6l4-6V3ZM5 14h14M9 7h6"/><path d="M9 17h1m4 1h1"/>',
  screen: '<rect x="3" y="3" width="18" height="14" rx="2"/><path d="M8 21h8m-4-4v4M7 7h4m-4 4h2"/>',
  log: '<path d="M7 3h10l3 3v15H4V3h3Zm8 0v5h5M8 12h8m-8 4h8"/>',
  settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="currentColor" stroke="none"/><circle cx="15" cy="17" r="3" fill="currentColor" stroke="none"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/>',
  scan: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M4 12h16"/>',
  play: '<path d="m7 4 14 8-14 8V4Z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="1"/>',
  refresh: '<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5m-4 8a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.grid}</svg>`;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });
let state = null, busy = false, connected = false, generation = 0, featureKey = '', detailId = '', customKey = '', eventKey = '', groupId = '', portalDirty = false;
let settingsDirty = false, detailDirty = false, scheduleDirty = false, toastTimer, currentId = null, frame = null, picture = null, selection = null, drag = null, loadingImage = false;
let deviceScan = null, deviceScanStarted = false, deviceAdbDirty = false, deviceScanning = false;
const canvas = $('canvas'), ctx = canvas.getContext('2d');
/* 分组元数据由后端提供；子任务保留原 ID、开关和调度记录。 */
const taskGroups = () => state?.task_groups || [];
const groupFeatures = group => group.features.map(id => state.features.find(f => f.id === id)).filter(Boolean);
const effective = feature => feature.effective_enabled ?? feature.enabled;
const groupFor = feature => taskGroups().find(g => g.features.includes(feature.id));
const portalMode = () => state.features.find(f => f.id === 'stone_farm')?.enabled ? 'stone' : state.features.find(f => f.id === 'parallel_world')?.enabled ? 'legacy' : 'off';
const stonePairText = () => Number.isInteger(state.stone?.start_stage) && Number.isInteger(state.stone?.final_stage) ? `${state.stone.start_stage}→${state.stone.final_stage}阶` : '等待识别当前阶段';
const portalLabel = () => ({stone:`降阶刷最后BOSS · ${stonePairText()}`,legacy:'旧版任务练级',off:'循环关闭'}[portalMode()]);
const dailyTime = feature => ({guild_daily:'18:00',pass_daily:'10:00'}[feature.id] || '08:00');
function durationText(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return '尚未读取';
  for (const [unit,label] of [[86400,'天'],[3600,'小时'],[60,'分钟']]) {
    if (value >= unit && value % unit === 0) return `${value/unit}${label}`;
  }
  return `${value}秒`;
}

function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function toast(text) {
  clearTimeout(toastTimer); $('toast').textContent = text; $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}
async function api(path, data) {
  const options = {signal: AbortSignal.timeout(25000)};
  if (data !== undefined) Object.assign(options, {method:'POST', headers:{'Content-Type':'application/json','X-Session-Token':TOKEN}, body:JSON.stringify(data)});
  const response = await fetch(path, options), body = await response.json();
  if (response.status === 403) {
    // 面板每次重启都会换发会话令牌;旧页面携的是旧令牌,所有 POST 都会被拒。
    // 这里直接重载页面以取得新令牌,避免"点开始没反应"。
    notice('面板已重启，正在刷新页面重新获取令牌…');
    setTimeout(() => location.reload(), 600);
    throw Error(body.error || '面板已重启，请稍候');
  }
  if (!response.ok) throw Error(body.error || '请求失败');
  return body;
}
async function act(fn, success) {
  if (busy) return;
  busy = true; generation++; notice(''); updateControls();
  try { await fn(); await refreshStatus(); if (success) toast(success); }
  catch (error) { notice(error.message); }
  finally { busy = false; updateControls(); }
}
function checked(feature) { return feature.enabled ? 'true' : 'false'; }
function toggleMarkup(feature) {
  if (['stone_farm', 'parallel_world'].includes(feature.id)) return '<a class="button" href="#group/stone">选择循环模式</a>';
  return `<button class="switch" type="button" role="switch" aria-label="${esc(feature.name)}开关" aria-checked="${checked(feature)}" data-feature-switch="${esc(feature.id)}"></button>`;
}
function executionKind(feature) {
  return {realtime:'实时触发',timed:'定时任务',daily:'每日任务'}[feature.execution?.mode] || '实时触发';
}
function modeTag(feature) {
  const mode = feature.execution?.mode || 'realtime';
  return `<span class="mode-tag ${mode}">${executionKind(feature)}</span>`;
}
function nextRunText(feature) {
  const task = feature.execution || {};
  if (feature.id === 'meter_calibration' && state?.meter) {
    // 校准是条件触发:直接把"读到的角色等级/标尺/差距"和当前的等待原因显示出来,
    // 否则用户只能看到一行空白,无法判断是在等条件还是出故障。
    const meter = state.meter;
    const measured = meter.player_level != null && meter.value != null
      ? `角色 Lv${meter.player_level} · 标尺 ${meter.value} · 差距 ${meter.gap}(超过 ${meter.trigger_gap} 才校准)` : null;
    if (meter.recovery_required) return `本轮已跳过 · ${meter.hold || '只读等待实际钓鱼及原航巡恢复'}`;
    if (meter.pending) return `校准进行中 · ${measured || '正在核对画面与航巡状态'}`;
    if (meter.next_check > Date.now()/1000) return `${meter.wait_kind === 'failure_backoff' ? '上次未完成 · 重试等待' : '成功校准冷却'} · 剩余 ${Math.ceil((meter.next_check-Date.now()/1000)/60)} 分钟 · ${measured || '等待下次读数'}`;
    return measured ? `${measured}${meter.hold ? ` · ${meter.hold}` : ''}` : (meter.hold || '等待可读画面');
  }
  if (task.mode === 'realtime') return task.condition || '条件出现时处理';
  if (task.status === 'running') return '本轮执行中 · 完成后安排下一次';
  if (!task.next_run) return feature.enabled ? '已到期 · 等待画面条件满足' : '开启后检查';
  const at = new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai', month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(new Date(task.next_run));
  if (task.status === 'disabled') return `已关闭 · 保留时间 ${at}`;
  if (task.seconds_until <= 0) return `已到期 ${at} · 等待画面条件满足`;
  const seconds = Math.ceil(task.seconds_until);
  const countdown = seconds >= 3600 ? `${Math.floor(seconds/3600)}小时${Math.ceil(seconds%3600/60)}分` : seconds >= 60 ? `${Math.ceil(seconds/60)}分` : `${seconds}秒`;
  const outcome = task.reason === 'manual' ? '手动安排' : task.outcome === 'blocked' ? '等待安全条件 · 重查' : task.outcome === 'skipped' ? '本轮跳过 · 下次检查' : task.outcome === 'retry' || task.reason === 'retry' ? '上轮未完成 · 重试' : '下次开启';
  return `${outcome} ${at} · ${countdown}后${task.detail ? ` · ${task.detail}` : ''}`;
}
function timeInputValue(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  // 一律按北京时间渲染，避免把 UTC 当成当地时间（旧实现加 8 小时再取 UTC 串，
  // 遇到不带时区的兜底值会整体慢 8 小时）。
  return new Intl.DateTimeFormat('sv-SE', {timeZone:'Asia/Shanghai', year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hourCycle:'h23'}).format(date).replace(' ', 'T');
}

function scheduledFeatures(list) {
  return list.filter(f => effective(f) && f.available && f.execution?.mode !== 'realtime')
    .sort((a,b) => (Date.parse(a.execution?.next_run || '') || 0) - (Date.parse(b.execution?.next_run || '') || 0));
}
function currentFeature() {
  if (!state.running || !state.stage || state.stage === 'main') return null;
  const prefixes = {stone:'stone_farm',parallel:'parallel_world',equipment:'equipment_upgrade',skills:'parallel_skills',cruise:'parallel_skills',meter:'meter_calibration',backpack:'backpack_sell',guild_daily:'guild_daily',restaurant:'restaurant_upgrade',return_parallel:'return_parallel',pass:'pass_daily'};
  const key = Object.keys(prefixes).sort((a,b) => b.length-a.length).find(k => state.stage === k || state.stage.startsWith(k+'_'));
  if (key) return state.features.find(f => f.id === prefixes[key]);
  const rule = state.rules?.find(r => r.stage === state.stage);
  return rule ? state.features.find(f => f.rules.includes(rule.id)) : null;
}
function groupNext(group) {
  const list = groupFeatures(group), current = currentFeature();
  if (current && list.includes(current)) return `正在执行：${current.name}`;
  const scheduled = scheduledFeatures(list), realtime = list.filter(f => effective(f) && f.execution?.mode === 'realtime');
  const next = scheduled[0];
  return [next ? `${next.name} · ${nextRunText(next)}` : '', realtime.length ? `${realtime.length}项按实时条件触发` : ''].filter(Boolean).join('；') || '全部关闭';
}

/* ---------- 侧栏:八个业务入口 ---------- */
function renderMenu() {
  $('task-menu').innerHTML = taskGroups().map(group => `<a class="nav-item nav-task" href="#group/${esc(group.id)}" data-page="group/${esc(group.id)}">${icon(group.icon)}${esc(group.name)}<span class="nav-group-count" data-group-count="${esc(group.id)}"></span></a>`).join('');
}

/* ---------- 概览只展示合并后的入口，子项在组内设置 ---------- */
function renderGroups() {
  $('task-groups').innerHTML = taskGroups().map(group => `<a class="task-bundle" href="#group/${esc(group.id)}" data-group="${esc(group.id)}">
    <div class="bundle-heading"><span class="feature-icon">${icon(group.icon)}</span><h3>${esc(group.name)}</h3><span class="group-count" data-group-count="${esc(group.id)}"></span></div>
    <p>${esc(group.description)}</p><p class="bundle-next" data-group-next="${esc(group.id)}"></p><span class="row-link">管理子项 ${icon('chevron')}</span></a>`).join('');
}
function renderGroup(group) {
  if (groupId !== group.id) {
    groupId = group.id; portalDirty = false;
    const list = groupFeatures(group).filter(f => group.id !== 'stone' || !['stone_farm','parallel_world'].includes(f.id));
    const mode = group.id === 'stone' ? `<section class="panel"><h2>次元石循环模式</h2><form id="portal-mode-form"><label for="portal-mode">运行模式</label><select id="portal-mode"><option value="stone">降阶刷最后BOSS</option><option value="legacy">旧版任务练级</option><option value="off">关闭次元石循环</option></select><p id="portal-mode-state" class="detail-note" role="status"></p><button class="button primary" type="submit">保存循环模式</button><p class="field-help">停止脚本后可切换。只切换循环方式，返回平行世界和武器强化使用各自开关；进度、截止时间和已用次数保留。</p></form><div class="mode-links"><a href="#task/stone_farm">降阶流程与记录</a><a href="#task/parallel_world">旧版任务练级设置</a></div></section>` : '';
    $('group-detail').innerHTML = `<div class="page-heading"><div><div class="eyebrow">任务设置</div><h1>${esc(group.name)}</h1><p>${esc(group.description)}</p></div><a class="button" href="#overview">返回总览</a></div>${mode}<div class="task-rows">${list.map(taskRow).join('')}</div><p class="field-help">各子项独立保存开关和记录；实时任务按条件触发，定时与每日任务各自安排下次执行。</p>`;
    if ($('portal-mode-form')) {
      $('portal-mode').onchange = () => { portalDirty = true; };
      $('portal-mode-form').onsubmit = event => { event.preventDefault(); const mode=$('portal-mode').value; act(async () => { await api('/api/portal-mode',{mode}); portalDirty=false; },'循环模式已保存'); };
    }
  }
  if ($('portal-mode-form') && group.id === 'stone') {
    if (!portalDirty) $('portal-mode').value = portalMode();
    $('portal-mode-state').textContent = `已选模式：${portalLabel()}。` + (portalMode()==='stone' && state.features.find(f=>f.id==='parallel_world')?.enabled ? '旧模式的已保存开关尚在，当前不会执行；切换时会统一保存为单一模式。' : '两种循环模式互斥，武器强化仍按自己的间隔执行。');
  }
}
function taskRow(feature) {
  return `<div class="task-row${feature.enabled ? '' : ' off'}" data-card="${esc(feature.id)}">
    <div class="task-row-main">
      <span class="feature-icon ${esc(feature.id)}">${icon(feature.icon)}</span>
      <div class="task-row-text">
        <h3><a href="#task/${esc(feature.id)}">${esc(feature.name)}</a> ${modeTag(feature)}</h3>
        <p data-summary="${esc(feature.id)}"></p>
        <p class="task-next" data-next-run="${esc(feature.id)}"></p>
      </div>
      <div class="task-row-side">
        <span class="feature-status" data-feature-status="${esc(feature.id)}"></span>
        ${toggleMarkup(feature)}
        <a class="row-link" href="#task/${esc(feature.id)}" aria-label="${esc(feature.name)}设置">设置 ${icon('chevron')}</a>
      </div>
    </div>
  </div>`;
}
function renderFeatures() {
  const key = JSON.stringify(taskGroups()) + state.features.map(f => f.id).join(',');
  if (featureKey !== key) { featureKey = key; renderGroups(); renderMenu(); }
  const active = state.features.filter(f => f.enabled).length;
  $('enabled-count').textContent = `${taskGroups().length}个入口 · ${active}项已开启`;
  for (const group of taskGroups()) {
    const list = groupFeatures(group), on = list.filter(effective).length;
    document.querySelectorAll(`[data-group-count="${group.id}"]`).forEach(el => { el.textContent = `${on}项已启用`; });
    document.querySelectorAll(`[data-group-next="${group.id}"]`).forEach(el => { el.textContent = groupNext(group); });
  }
  for (const f of state.features) {
    const status = !f.available ? '规则缺失' : f.partial ? '部分开启 · 待调整' : f.inactive_reason || (!f.enabled ? '已关闭' : f.schedule ? (f.schedule.completed ? `本周期已完成 · 下个周期 ${dailyTime(f)} 后检查` : '已开启 · 本周期待完成') : `已开启 · ${executionKind(f)}`);
    document.querySelectorAll(`[data-feature-switch="${f.id}"]`).forEach(el => el.setAttribute('aria-checked', checked(f)));
    document.querySelectorAll(`[data-feature-status="${f.id}"]`).forEach(el => { el.textContent = status; el.className = 'feature-status' + (f.partial ? ' partial' : f.enabled ? '' : ' off'); });
    document.querySelectorAll(`[data-card="${f.id}"]`).forEach(el => el.classList.toggle('off', !f.enabled));
    document.querySelectorAll(`[data-nav-dot="${f.id}"]`).forEach(el => el.classList.toggle('on', f.enabled));
    const summary = f.id === 'mana' ? `法力低于 ${f.mana_threshold ?? '—'} 时补充 · 每轮一瓶` : f.summary;
    document.querySelectorAll(`[data-summary="${f.id}"]`).forEach(el => { el.textContent = summary; });
    document.querySelectorAll(`[data-next-run="${f.id}"]`).forEach(el => { el.textContent = nextRunText(f); });
  }
}

/* ---------- 概览:调度器状态 ---------- */
function renderScheduler() {
  const labels = {idle:'未运行',preview:'识别预览中',clicking:'自动运行中',stopping:'正在停止'};
  const running = !!state.running;
  $('mode').textContent = labels[state.mode] || state.mode;
  $('status-light').dataset.state = running ? 'on' : 'off';
  const headerMode = $('header-mode');
  headerMode.textContent = labels[state.mode] || state.mode;
  headerMode.dataset.state = state.mode === 'stopping' ? 'warn' : running ? 'on' : 'off';
  $('instance-icon').dataset.state = running ? 'on' : 'off';
  const active = state.features.filter(effective);
  const scoped = state.scoped ? state.features.filter(f => (state.active_rule_ids || []).some(id => f.rules.includes(id))) : null;
  const current = currentFeature();
  const pending = scheduledFeatures(scoped || active).filter(f => f !== current);
  $('running-task').textContent = current ? `${groupFor(current)?.name || current.category} / ${current.name}` : (state.running ? state.stage === 'game_recovery' ? '后台恢复 / 游戏重连' : '识别主画面' : '—');
  $('pending-task').textContent = pending[0] ? `${pending[0].name} · ${nextRunText(pending[0])}` : '暂无定时任务';
}
const notes = {
  restaurant_upgrade:'在当前世界的餐馆进入料理图鉴，强化餐品的永久效果。按所选类别检查已解锁餐品，长按交付四种食材；每升一级重新核对，材料充足时继续强化，直到满级或无法继续交付。缺料餐品跳过，已有交付进度保留。结束后恢复原地点或原航巡，再安排下次执行。此功能对应图鉴强化，不是制作或食用料理。',
  guild_daily:'每天北京时间18:00执行。公会任务中的“已完成”表示指标已达成，可以提交；逐项点击后确认任务行变化，最终暗色“领取奖励”才属于已领取状态。检查紫色和蓝色两类，保留原世界、航巡和钓鱼状态。直达公会消耗10枚蓝色公会币，余额不足时取消并恢复原活动，不使用钻石。',
  parallel_skills:'启用本项后，分别管理游戏航巡和巡航系技能。保存新设置后等待当前任务收尾，再同步游戏选项。航巡里的自动贩卖会消耗出售券，低于阈值时用药水补票；自动强化是游戏航巡选项，脚本定时的鱼饼强化在“次元石循环”中单独设置。三种技能分别控制自动续与技能强化，只在平行世界检查；经验加速到期后仅用技能点购买。',
  task_popup:'处理已识别的任务与每日刷新弹窗。出现弹窗时先让当前任务暂停点击，关闭后重新核对所在页面，再接续原流程；不把弹窗消失当成原任务已完成。',
  update_prompt:'识别游戏更新询问后选择取消，确认提示消失，再接续当前任务。',
  relief:'仅识别黄色水滴与浅色星形两种救济品。降落伞位置变化时，按当前画面的位置点击。',
  mana:'法力严格小于设定值才会触发；等于阈值时跳过。库存需大于 0，每轮使用一瓶，随后关闭技能面板。',
  achievements:'只领取绿色按钮。每次领取后重新识别，直到没有绿色按钮，再自动关闭列表、收起菜单。',
  mailbox:'北京时间08:00刷新后，有新邮件提示时打开邮箱并全部领取。确认邮箱已空、返回主界面后记录本周期完成。物品达到获取上限时关闭邮箱并保留邮件，15分钟后再检查；本周期领空后，即使出现新邮件也等下一次刷新。',
  daily_reward:'北京时间08:00刷新后进入PREMIUM页，分别检查免费奖励和右侧附赠奖励。附赠明亮可领时领取，灰色时跳过；完成记录分别保存，返回主界面后等待下次刷新。',
  pass_daily:'每日08:00重置，10:00后检查钓鱼进度；未满120条时关闭并等待重试。达到120条后点击免费奖励，确认30/60/90/120四档均已领取，再记录本周期完成，等待下一周期10:00。',
  event_potion:'当期活动材料连续三帧读数一致且严格大于 500 时，打开活动商店，只购买第一个标价 500 的药水。确认购买后点击红叉返回主界面。每轮买一瓶，默认间隔 60 秒；下一轮仍超过 500 才继续，等于 500 或读不清时跳过。',
  guild_shop:'每天北京时间 08:00 刷新。进入 SPECIAL 公会点数页购买药水和天赋值，逐次加一至可购买上限，再核对数量、货币和金额后确认。药水每份 10 点、最多 10 份；天赋值每份 50、20 点、最多 5 份。两种商品均确认买满且返回主界面后等待下次刷新；点数不足或未买满保留重试。完成记录重启后保留，离线错过刷新时恢复运行后补做。',
  raid:'出现可参加的突袭提示后进入并准备，等待可开始时参战。进入场地后，若BOSS未出现则点击中央圆盘抛竿；一旦确认真实战斗就停止补点。战斗结束后确认结算，或核对游戏已自动返回主界面，再接续原任务。',
  backpack_sell:'此开关控制脚本前往港口出售，与“巡航设置”中的游戏自动贩卖相互独立。确认钓鱼材料背包已满后，先停航并保存原地点，再到当前世界港口出售第一标签的未锁定材料。料理及锁定物品保留；确认容量后返回原地点，并按已有记录接续活动。',
  equipment_upgrade:'五种武器各检查一次，绿色鱼饼按钮可强化时长按一次，完成一遍就关闭面板。实际战斗期间暂缓打开；已确认入口未展开时等待下个检查间隔。此开关与游戏航巡中的自动强化分别控制，关闭次元石循环也不会自动关闭本项。',
  meter_calibration:'在平行世界普通地点校准，保留航巡，仅在普通鱼准备阶段停止持续钓鱼。确认无填充、无战斗后，读取当前范围并调整到角色等级附近；途中换地点则重新读取。长按启动后，以连续新图的真实填充或战斗确认恢复。任务BOSS、港口或圆球AUTO未开启时等待；已有填充或战斗时不重复拖动和长按。',
  stone_farm:'游戏航巡负责钓鱼、升级与尝试BOSS。连续确认当前阶段后，RANK39先安全停航停钓，再返回观测所降一阶，保留本轮进度打最后一关（例如10→9）。降阶后的世界实际RANK40后安全返回，确认旧门关闭，再打开本轮原阶段的新门并启动钓鱼和航巡。旧交接按记录的原阶段继续，未知或冲突数字等待重认。恢复航巡时使用已启用的巡航设置；未启用管理时沿用兼容路线。守护者补伤害、海神反击与补伤害按相应已保存开关执行。此模式与旧版任务导航练级互斥。',
  parallel_world:'自动寻找骷髅任务点。普通 BOSS 实际钓鱼失败达到设定次数后开始练级；成功会清除失败计数。最后一个 BOSS（RANK39→40）低于单独设定的挑战等级时先练级，默认达到 BOSS 同级再挑战。默认开启练级自动航行，只勾选珍珠、琥珀、蓝色宝石，取消天赋材料和 BOSS；不启用游戏内自动出售或自动强化。航行期间继续已开启的钓鱼加速和鱼饼武器强化。达到任务 BOSS 等级减去对应允许差值后，先确认航行停止，再返回试打；时间或本轮升级数量不会提前触发。关闭自动航行时，仍可前往其他红色地点练级；连续确认未钓鱼后调整计量并长按圆盘，看到浅蓝色填充才确认启动，已有填充或抛竿、收鱼时不重复操作。练级进度重启后保留。连续三次确认 RANK 40 后返回并重开次元门。',
  growth_ad:'每日额度读取游戏右上角计数；满额或读不清时跳过。不会因为重启脚本重置额度。',
  potion_ad:'每日额度读取游戏右上角计数；满额或读不清时跳过。不会因为重启脚本重置额度。'
};
function featureNote(feature) {
  const note = notes[feature.id] || feature.description;
  if (feature.id === 'equipment_upgrade') return `当前检查间隔${durationText(feature.interval)}，每种武器长按${feature.hold_seconds ?? '—'}秒。${note}`;
  if (feature.id === 'restaurant_upgrade') return `当前检查间隔${durationText(feature.interval)}。${note}`;
  if (feature.id === 'meter_calibration') {
    const c = feature.calibration || {}, gap = c.trigger_gap ?? state?.meter?.trigger_gap ?? '尚未读取';
    return `当前触发差距：严格超过${gap}级；成功冷却${durationText(feature.interval)}，目标允许误差${c.tolerance ?? '—'}级。${note}`;
  }
  return note;
}
function cruiseFields(feature) {
  if (feature.id !== 'parallel_skills' || !feature.cruise_settings) return '';
  const c = feature.cruise_settings.cruise, s = feature.cruise_settings.skills;
  const check = (id, label, value) => `<label class="cruise-check"><input type="checkbox" id="${id}" ${value ? 'checked' : ''}>${label}</label>`;
  return `<fieldset class="cruise-section"><legend>巡航</legend>${check('cruise-enabled','管理游戏巡航',c.enabled)}<div class="cruise-grid">${[['pearl','珍珠'],['material','技能（天赋材料）'],['amber','琥珀'],['gem','蓝色宝石'],['boss','BOSS']].map(([key,label])=>check('cruise-target-'+key,label,c.targets[key])).join('')}</div><div class="cruise-grid">${check('cruise-sale','游戏自动贩卖',c.auto_sale)}${check('cruise-upgrade','游戏自动强化',c.auto_upgrade)}</div><label for="cruise-ticket-min">自动贩卖券低于此数量时补票</label><input type="number" id="cruise-ticket-min" min="0" max="9999" step="1" value="${c.ticket_min}" required><p class="field-help">启用巡航管理及游戏自动贩卖时检查；0表示不补票。用药水长按＋号补充，每次核对数量；保持游戏当前的目的地优先顺序。</p></fieldset><fieldset class="cruise-section"><legend>巡航系技能</legend>${check('cruise-skills-enabled','管理平行世界技能',s.enabled)}<div class="cruise-skill-table"><span>技能</span><span>自动续</span><span>技能强化</span>${[['fishing','钓鱼加速'],['movement','移动加速'],['time','时间加速']].map(([key,label])=>`<span>${label}</span>${check('skill-'+key+'-auto',label+'自动续',s[key].auto)}${check('skill-'+key+'-enhance',label+'强化',s[key].enhance)}`).join('')}</div>${check('cruise-growth','经验buff：仅用技能点购买',s.growth_points)}<p class="field-help">50%成长加速，200技能点/小时。已有加速或有效购买记录时不重复购买。</p></fieldset><p id="cruise-apply-state" class="detail-note" role="status"></p>`;
}
function readCruiseFields() {
  const on = id => $(id).checked;
  return {cruise:{enabled:on('cruise-enabled'),targets:Object.fromEntries(['pearl','material','amber','gem','boss'].map(k=>[k,on('cruise-target-'+k)])),auto_sale:on('cruise-sale'),auto_upgrade:on('cruise-upgrade'),ticket_min:Number($('cruise-ticket-min').value)},skills:{enabled:on('cruise-skills-enabled'),...Object.fromEntries(['fishing','movement','time'].map(k=>[k,{auto:on('skill-'+k+'-auto'),enhance:on('skill-'+k+'-enhance')}])),growth_points:on('cruise-growth')}};
}
function renderDetail(feature) {
  if (feature.id === 'return_parallel') {
    if (detailId !== feature.id) {
      detailId=feature.id; detailDirty=false; scheduleDirty=false;
      $('feature-detail').innerHTML=`<div class="page-heading"><div><div class="eyebrow">次元石循环 · 实时触发</div><h1>${esc(feature.name)}</h1><p>${esc(feature.description)}</p></div><a class="button" href="#group/stone">返回次元石循环</a></div><section class="panel"><div class="feature-enable"><h2>自动返回平行世界</h2>${toggleMarkup(feature)}</div><p class="detail-note">普通世界停船且满足进入条件时检查；不设置定时日期。开关独立于循环模式，已有返回进度保留。</p><ol class="steps">${feature.steps.map(step=>`<li>${esc(step)}</li>`).join('')}</ol></section>`;
    }
    return;
  }
  if (detailId !== feature.id) {
    detailId = feature.id; detailDirty = false; scheduleDirty = false;
    const restaurantCategories = ['干货', '炖煮', '生鱼片', '油炸', '炒饭', '面食', '寿司', '汤', '烧烤'];
    const restaurantEffects = {'干货':'鱼饼获得量', '炖煮':'鱼竿攻击力', '生鱼片':'鱼叉塔攻击力', '油炸':'大炮攻击力', '炒饭':'机关炮塔攻击力', '面食':'魔法石攻击力', '寿司':'鲸鱼攻击力', '汤':'琥珀/海蓝宝/次元石获得量', '烧烤':'经验值获得量'};
    const restaurantLabel = name => name ? `${name}（${restaurantEffects[name] || ''}）` : '全部';
    const restaurantSelection = feature.targets?.length === restaurantCategories.length ? '' : (feature.targets?.[0] ?? '');
    const restaurantTargets = feature.id === 'restaurant_upgrade' ? `<label for="restaurant-targets">强化类别</label><select id="restaurant-targets" name="targets" style="width:100%;padding:10px 12px;border:1px solid #dce4e9;border-radius:6px;background:#fff;color:#536b7a;font-size:13px">${['', ...restaurantCategories].map(name => `<option value="${name}" ${name === restaurantSelection ? 'selected' : ''}>${restaurantLabel(name)}</option>`).join('')}</select><p class="field-help">选择“全部”时检查九类已解锁餐品，选择单一类别时只检查该类。材料充足可连续升多级；无法继续交付或满级时换下一种，结束后恢复原活动。</p>${feature.targets?.length > 1 && restaurantSelection ? '<p class="field-help">原设置包含多个类别，当前显示首个类别；保存后仅强化下拉选中的类别。</p>' : ''}` : '';
    $('feature-detail').innerHTML = `<div class="page-heading"><div><div class="eyebrow">${esc(feature.category)} · ${executionKind(feature)}</div><h1>${esc(feature.name)}</h1><p>${esc(feature.description)}</p></div><a class="button" href="#overview">返回任务组</a></div><div class="detail-layout"><div><section class="panel"><div class="feature-enable"><span class="feature-icon ${esc(feature.id)}">${icon(feature.icon)}</span><div><h2>启用${esc(feature.name)}</h2><p>开关即时保存，独立控制本项流程。</p></div>${toggleMarkup(feature)}</div><div class="detail-note">${esc(featureNote(feature))}</div>${feature.schedule ? '<p class="detail-note" id="daily-schedule" role="status"></p>' : ''}</section><section class="panel"><h2>功能参数</h2><form id="feature-form" class="parameter-form"><label for="feature-interval">${feature.schedule ? '未完成时重试间隔 / 秒' : '再次触发间隔 / 秒'}</label><input id="feature-interval" type="number" min="0.2" max="3600" step="0.1" required><p class="field-help">${feature.schedule ? '完成后等待下个周期的领取时间；未完成时按保存间隔重试。' : '完成操作后，至少等待此间隔才再次触发同一入口。'}</p>${feature.id === 'mana' ? '<label for="mana-threshold">法力低于此值时补充</label><input id="mana-threshold" type="number" min="1" max="100" required><p class="field-help">默认 25。数字读取不清晰时不会使用药水。</p>' : ''}${restaurantTargets}${cruiseFields(feature)}${feature.daily_limit ? `<label>游戏每日领取上限</label><p class="detail-note">${feature.daily_limit} 次 · 以游戏显示计数为准</p>` : ''}<button class="button primary form-submit" id="save-feature" type="submit">保存功能参数</button><p id="parameter-apply-state" class="detail-note" role="status"></p><p class="field-help">支持运行中保存；当前任务完成并返回主画面后应用，原截止时间、已用次数和购买记录保留。</p></form></section></div><section class="panel"><h2>执行流程</h2><ol class="steps">${feature.steps.map(step => `<li>${esc(step)}</li>`).join('')}</ol><details><summary>查看 ${feature.rules.length} 条识别规则</summary><div id="detail-rules"></div></details></section></div>`;
    const realtime = feature.execution?.mode === 'realtime';
    if (feature.id === 'stone_farm') {
      $('feature-form').closest('.panel').insertAdjacentHTML('beforebegin', '<section class="panel"><h2>交接记录</h2><p id="stone-record-state" class="detail-note" role="status"></p><button class="button" id="stone-clear" type="button">清除交接记录</button><p class="field-help">停止脚本后可清除。原记录先归档，保留输入次数；只解除本次交接占用，不操作游戏或自动重做同一交接。过期交接在下次开始时自动归档。</p></section>');
      $('stone-clear').onclick = () => act(() => api('/api/stone-clear', {sha256: state.stone_record.sha256}), '交接记录已归档，待处理状态已清除');
    }
    if (feature.id === 'meter_calibration') {
      $('feature-form').closest('.panel').insertAdjacentHTML('beforebegin', '<section class="panel"><h2>标尺记录</h2><p id="meter-record-state" class="detail-note" role="status"></p><button class="button" id="meter-clear" type="button">清除标尺记录</button><p class="field-help">停止脚本后可清除。原记录先归档，保留截止时间、输入次数和范围缓存。清除不操作游戏，不追加输入；重新启动后只读核对实际钓鱼和原航巡恢复，再按间隔检测。过期执行在下次开始时自动归档。</p></section>');
      $('meter-clear').onclick = () => act(() => api('/api/meter-clear', {sha256: state.meter_record.sha256}), '标尺执行已归档，将只读核对恢复状态');
    }
    $('feature-form').closest('.panel').insertAdjacentHTML('beforebegin', `<section class="panel schedule-panel"><h2>${executionKind(feature)}</h2><p id="task-next-run" class="detail-note" role="status"></p>${realtime ? '<p class="field-help">持续检测当前画面，条件满足时处理；防重复间隔只限制再次触发，不设置下次开启日期。</p>' : '<form id="schedule-form"><label for="task-next-time">下次开启时间 / 北京时间</label><input id="task-next-time" type="datetime-local" step="1" required><p class="field-help">时间到后仍须满足画面条件；下次开启时间和完成记录在重启后保留。</p><div class="schedule-actions"><button class="button primary" type="submit">保存时间</button><button class="button" id="task-ready" type="button">设为可执行</button></div><p class="field-help">运行中可修改未在执行的任务；不会打开已关闭的任务或重置正在进行的任务。</p></form>'}</section>`);
    if (realtime) {
      document.querySelector('label[for="feature-interval"]').textContent = '防止重复触发间隔 / 秒';
      $('feature-interval').nextElementSibling.textContent = '持续检测触发条件；本次处理后，至少等待此间隔才再次触发同一入口。';
      if (feature.id === 'meter_calibration') {
        $('feature-interval').max = '86400';
        document.querySelector('label[for="feature-interval"]').textContent = '成功校准后冷却 / 秒';
        $('feature-interval').nextElementSibling.textContent = `此间隔仅用于成功校准。当前失败退避为${durationText(feature.calibration?.failure_backoff)}，确认实际恢复后再检查；仍需严格超过${feature.calibration?.trigger_gap ?? '—'}级差距。重启保留冷却与已用次数。`;
      }
    } else if (feature.id === 'guild_daily') {
      $('feature-interval').max = '86400';
      $('feature-interval').nextElementSibling.textContent = '每天北京时间18:00执行；完成后等待次日18:00。未完成保留记录，不把灰色按钮当作已提交。';
      document.querySelector('#schedule-form .field-help').textContent = '每日18:00后才可执行，重启保留当天完成记录和下次开启时间。';
    } else if (!feature.schedule) {
      document.querySelector('label[for="feature-interval"]').textContent = '完成后间隔 / 秒';
      if (feature.id === 'restaurant_upgrade') {
        $('feature-interval').max = '259200';
        $('feature-interval').nextElementSibling.textContent = '当前值见上方输入框。本轮结束并恢复原活动后开始计时；材料充足时可在同一轮连续强化。';
      }
    }
    if (feature.schedule) {
      $('feature-interval').nextElementSibling.textContent = `每天北京时间08:00重置，本项${dailyTime(feature)}后检查。完成后等待下一周期；未完成按此间隔重试，仍需满足领取条件。`;
      document.querySelector('#schedule-form .field-help').textContent = `下次开启时间可推迟检查；本项仍须到${dailyTime(feature)}并满足画面条件。本周期已完成时等待下一周期，记录在重启后保留。`;
    }
    if ($('schedule-form')) {
      if (!scheduleDirty) $('task-next-time').value = timeInputValue(feature.schedule?.next_run || feature.execution?.next_run) || timeInputValue(new Date());
      $('schedule-form').addEventListener('input', () => { scheduleDirty = true; });
      $('schedule-form').onsubmit = event => { event.preventDefault(); act(async () => { await api('/api/tasks/schedule', {id: feature.id, next_run: new Date($('task-next-time').value + '+08:00').toISOString()}); scheduleDirty = false; }, '下次开启时间已保存'); };
      $('task-ready').onclick = () => act(async () => { await api('/api/tasks/schedule', {id: feature.id, next_run: new Date(Date.now()).toISOString()}); scheduleDirty = false; }, '已设为可执行');
    }
    $('feature-form').addEventListener('input', () => { detailDirty = true; });
    const interval = $('feature-interval');
    if (feature.id === 'parallel_skills') {
      document.querySelector('label[for="feature-interval"]').textContent = '定期技能与补票检查间隔 / 秒';
      interval.nextElementSibling.textContent = '保存新设置会立即排队同步；定期检查按此间隔执行，补票检查最短60秒。';
    }
    interval.value = feature.interval ?? 1;
    interval.addEventListener('input', () => { detailDirty = true; });
    if ($('restaurant-targets')) $('restaurant-targets').addEventListener('change', () => { detailDirty = true; });
    if ($('mana-threshold')) { $('mana-threshold').value = feature.mana_threshold ?? 25; $('mana-threshold').addEventListener('input', () => { detailDirty = true; }); }
    $('feature-form').onsubmit = event => {
      event.preventDefault();
      const data = {id: feature.id, interval: Number(interval.value)};
      if ($('mana-threshold')) data.mana_threshold = Number($('mana-threshold').value);
      if (feature.id === 'parallel_world' && feature.training) data.training = feature.training;
      if (feature.id === 'parallel_skills') { if ($('cruise-enabled')) data.cruise_settings = readCruiseFields(); else data.growth_points = feature.growth_points; }
      if (feature.id === 'restaurant_upgrade') {
        const target = $('restaurant-targets').value;
        data.targets = target ? [target] : [];
      }
      act(async () => { await api('/api/features/change', data); detailDirty = false; }, '功能参数已保存，当前任务结束后应用');
    };
  }
  if ($('parameter-apply-state')) {
    $('parameter-apply-state').textContent = detailDirty ? '有未保存的修改，点击保存后应用。' : state.settings_pending ? '新参数已保存，等待当前任务结束并返回主画面后应用。' : '当前参数已生效。';
  }
  if ($('cruise-apply-state')) {
    $('cruise-apply-state').textContent = state.settings_pending ? '新设置已保存，等待当前操作结束后同步。' : state.cruise_setup?.hold || '保存后在下一个安全画面核对游戏设置。';
    if (!detailDirty) {
      const c=feature.cruise_settings.cruise, s=feature.cruise_settings.skills;
      for (const [id,value] of Object.entries({'cruise-enabled':c.enabled,'cruise-sale':c.auto_sale,'cruise-upgrade':c.auto_upgrade,'cruise-skills-enabled':s.enabled,'cruise-growth':s.growth_points})) $(id).checked=value;
      for (const [key,value] of Object.entries(c.targets)) $('cruise-target-'+key).checked=value;
      for (const name of ['fishing','movement','time']) for (const field of ['auto','enhance']) $('skill-'+name+'-'+field).checked=s[name][field];
      $('cruise-ticket-min').value=c.ticket_min;
    }
  }
  const note = document.querySelector('#feature-detail .feature-enable + .detail-note');
  if (note) note.textContent = featureNote(feature);
  if (feature.id === 'meter_calibration' && !detailDirty) {
    $('feature-interval').nextElementSibling.textContent = `此间隔仅用于成功校准。当前失败退避为${durationText(feature.calibration?.failure_backoff)}，确认实际恢复后再检查；仍需严格超过${feature.calibration?.trigger_gap ?? '—'}级差距。重启保留冷却与已用次数。`;
  }
  $('task-next-run').textContent = nextRunText(feature);
  if ($('stone-clear')) {
    const record = state.stone_record || {};
    $('stone-clear').disabled = busy || state.running || !record.pending || !!record.error;
    $('stone-record-state').textContent = `${stonePairText()} · ` + (record.error || (record.pending ? `${record.expired ? '已过期' : '待处理'} · ${record.phase}` : record.retirement ? '旧交接已归档，等待当前游戏进度更新' : '没有待处理的交接'));
  }
  if ($('meter-clear')) {
    const record = state.meter_record || {};
    $('meter-clear').disabled = busy || state.running || !record.pending || !!record.error;
    $('meter-record-state').textContent = record.error || (record.pending ? `${record.expired ? '已过期' : '待处理'} · 原截止和次数保留` : record.recovery_required ? '本轮已归档，只读等待实际钓鱼和原航巡恢复' : record.retirement ? '旧失败已归档，已确认实际恢复，按间隔检测' : '没有待处理的标尺执行');
  }
  if ($('daily-schedule')) $('daily-schedule').textContent = feature.schedule?.completed ? `本周期已完成 · ${feature.schedule.next_run ? nextRunText(feature) : '等待下次刷新'}` : '';
  $('detail-rules').innerHTML = feature.rules.map(key => {
    const rule = state.rules.find(r => r.id === key);
    return `<div class="rule-line"><span>${esc(feature.id === 'stone_farm' && key === 'driftinggirl_stone_farm' ? feature.name : (rule?.name || key))}</span><small>${rule ? (rule.enabled ? '启用' : '关闭') : '缺失'}</small></div>`;
  }).join('');
}
function route() {
  if (!state) return;
  const page = location.hash.slice(1) || 'overview';
  const feature = state.features.find(f => page === 'task/' + f.id);
  const group = taskGroups().find(g => page === 'group/' + g.id);
  const parent = feature ? groupFor(feature) : group;
  const pages = {overview:'任务总览',screen:'画面与识别',logs:'运行日志',settings:'运行设置'};
  const title = feature ? `${parent?.name || feature.category} / ${feature.name}` : group ? group.name : (pages[page] || pages.overview);
  $('crumb').textContent = title;
  for (const el of document.querySelectorAll('.page')) el.hidden = el.id !== 'page-' + (feature ? 'feature' : group ? 'group' : (pages[page] ? page : 'overview'));
  document.querySelectorAll('[data-page]').forEach(el => {
    const selected = el.dataset.page === (parent ? 'group/'+parent.id : page); el.classList.toggle('active', selected);
    if (selected) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
  });
  if (feature) {
    renderDetail(feature);
    const back = document.querySelector('#feature-detail .page-heading>a');
    if (back && parent) { back.href = '#group/'+parent.id; back.textContent = '返回'+parent.name; }
  } else detailId = '';
  if (group) { renderGroup(group); renderFeatures(); }
  else { groupId=''; portalDirty=false; $('group-detail').replaceChildren(); }
  updateControls();
  if (page === 'settings' && !deviceScanStarted && connected && !busy) {
    deviceScanStarted = true;
    scanDevices();
  }
}
window.addEventListener('hashchange', route);

function renderCustom() {
  const builtIn = new Set(state.features.flatMap(f => f.rules));
  const rules = state.rules.filter(r => !builtIn.has(r.id)), key = JSON.stringify(rules);
  if (key === customKey) return;
  customKey = key;
  $('custom-rules').innerHTML = rules.length ? rules.map(r => `<div class="custom-rule"><div class="rule-line"><span>${esc(r.name)}</span><button class="switch" role="switch" aria-label="${esc(r.name)}开关" aria-checked="${r.enabled}" data-custom-switch="${esc(r.id)}"></button></div><div class="rule-line"><span class="muted">阈值 ${esc(r.threshold)} · 间隔 ${esc(r.interval)}s</span><button class="delete-button" data-delete="${esc(r.id)}" aria-label="删除${esc(r.name)}">删除</button></div></div>`).join('') : '<p class="empty-text">暂无自定义目标。内置任务可直接在任务总览中开启。</p>';
}
function renderEvents() {
  const key = JSON.stringify(state.events);
  if (eventKey === key) return;
  eventKey = key;
  const row = e => `<div class="log-line"><time>${esc(e.time)}</time><span>${esc(e.message)}</span></div>`;
  const markup = state.events.map(row).join('') || '<p class="empty-text">任务启动后会在这里显示活动。</p>';
  for (const [id, toggle] of [['recent-events', $('log-scroll')], ['logs', $('log-scroll-full')]]) {
    const pane = $(id);
    if (!pane) continue;
    const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 60;
    pane.innerHTML = markup;
    if (atBottom && toggle?.checked !== false) pane.scrollTop = pane.scrollHeight;
  }
}
function updateControls() {
  const running = !!state?.running, unavailable = busy || !connected;
  $('run').disabled = unavailable || running || !state?.rules.some(r => r.enabled);
  $('preview').disabled = unavailable || running || !state?.rules.some(r => r.enabled);
  $('stop').disabled = unavailable || !running || state?.mode === 'stopping';
  if ($('stone-clear')) $('stone-clear').disabled = unavailable || running || !state?.stone_record?.pending || !!state?.stone_record?.error;
  if ($('meter-clear')) $('meter-clear').disabled = unavailable || running || !state?.meter_record?.pending || !!state?.meter_record?.error;
  $('refresh').disabled = unavailable || running; $('refresh-mini').disabled = unavailable || running;
  $('device-detect').disabled = unavailable || deviceScanning;
  $('device-adb-path').disabled = unavailable || running || deviceScanning;
  $('device-select').disabled = unavailable || running || deviceScanning || !deviceScan?.devices.length;
  $('device-connect').disabled = unavailable || running || deviceScanning || !deviceScan || !$('device-select').value;
  $('device-auto').disabled = unavailable || running || deviceScanning || !deviceScan?.automatic_available;
  document.querySelectorAll('[data-feature-switch]').forEach(el => {
    const f = state?.features.find(f => f.id === el.dataset.featureSwitch);
    el.disabled = unavailable || !f?.available || (running && state.scoped);
    if (f) el.setAttribute('aria-checked', checked(f));
  });
  document.querySelectorAll('#portal-mode-form select,#portal-mode-form button').forEach(el => { el.disabled = unavailable || running; });
  document.querySelectorAll('#schedule-form input,#schedule-form button,#feature-form input,#feature-form select,#feature-form button,#settings-form input,#settings-form button,#target-form input,#target-form button,[data-custom-switch],[data-delete]').forEach(el => { const scoped = !!state?.scoped; const scheduling = el.closest('#schedule-form'); const activeTask = state?.features.find(f => f.id === detailId)?.execution?.status === 'running'; const hot = el.closest('#settings-form') || (el.closest('#feature-form') && (!scoped || detailId === 'parallel_skills')) || (scheduling && !scoped && !activeTask); el.disabled = unavailable || (running && !hot); });
}
function showNumbers() {
  $('numbers').replaceChildren();
  for (const value of Object.values(state.numbers || {}).filter(v => v.frame === state.frames)) {
    const el = document.createElement('p');
    const reading = value.value === null ? '未读清' : value.kind === 'purchase_stock' ? `${value.value}/${value.limit} · 本次可买 ${value.target ?? '—'} 份 · 点数 ${value.balance_text || '未读清'}` : value.kind === 'purchase_quantity' ? `${value.value}/${value.limit ?? '—'} · 总价 ${value.price ?? '未读清'} 点` : value.kind === 'purchase_confirmation' ? `${value.value} 点 · 应付 ${value.limit ?? '未读清'} 点` : value.kind === 'daily_counter' ? `${value.value}/${value.limit}` : value.value;
    el.textContent = `${value.label}：${reading} · ${value.passed ? '条件满足' : '跳过：' + (value.skip_reason || '条件未满足')}`;
    $('numbers').append(el);
  }
}
async function refreshStatus() {
  const requestGeneration = generation, result = await api('/api/status');
  if (requestGeneration !== generation) return;
  state = result; connected = true;
  $('app-version').textContent = `v${state.version}`;
  $('app-version').title = `upstream ${state.upstream}`;
  $('app-version').hidden = !state.version;
  $('connection').dataset.state = 'on'; $('connection').innerHTML = '<i></i>已连接';
  const activeCount = state.features.filter(f => f.enabled).length;
  const stageRule = state.rules.find(r => r.stage === state.stage);
  const training = state.parallel?.training;
  const fishing = state.parallel?.fishing;
  const sailing = state.parallel?.sailing;
  const sailingPhases = {opening:'打开航行设置',configure:'勾选练级资源',await_active:'确认航行启动',await_off:'确认航行停止',await_config_off:'暂停旧航行',wake_stop:'检测到睡眠，准备停止航行',wake_wait_off:'睡眠恢复：确认航行停止',wake_prepare:'睡眠恢复：调整计量并启动钓鱼'};
  const sailingText = sailing?.phase !== 'idle' && sailing?.phase ? ` · ${sailingPhases[sailing.phase] || '检查航行状态'}` : sailing?.running ? ' · 自动航行：珍珠、琥珀、蓝色宝石' : '';
  const fishingPhases = {revealing:'移开数字框，读取计量范围',adjusting:'调整计量',starting:'等待大球浅蓝填充',active:'钓鱼中',unknown:'等待圆盘识别',detect:'检查计量'};
  const fishingText = (fishing?.value != null ? ` · 计量 ${fishing.value} / 目标 ${fishing.target} · ${fishingPhases[fishing.phase] || '检查画面'}` : '') + (fishing?.dial_state === 'sleeping' ? ' · 圆盘休眠 Zzz' : fishing?.blue_fill ? ' · 已看到浅蓝填充' : '');
  const parallelText = (training?.mode === 'farming' ? `平行世界 RANK ${state.parallel.rank ?? '—'} · 钓鱼练级 Lv${training.player_level ?? '—'} / BOSS Lv${training.boss_level ?? '—'} · 达到 Lv${training.retry_level ?? '待识别'} 后试打` : training?.mode === 'trial' ? `平行世界 RANK ${state.parallel.rank ?? '—'} · 返回 BOSS 试打 · 角色 Lv${training.player_level ?? '—'}` : `平行世界 RANK ${state.parallel?.rank ?? '—'} · 等待游戏自动战斗`) + sailingText + fishingText + (state.parallel?.defense?.battle_tap ? ` · 补伤害点击 ${state.parallel.defense.damage_taps || 0} 次` : '');
  const bagPhases = {idle:'继续钓鱼',prepare:'确认自动航行停止',resume:'恢复返回步骤',find_port:'选择附近港口',close_port_map:'前往港口',travel_port:'前往港口',bag:'检查保管箱',verify_sale:'确认出售结果',open_return:'准备返回原地点',find_origin:'选择原钓鱼地点',close_origin_map:'返回原地点',travel_origin:'返回原地点'};
  const bagText = `满背包出售 · ${bagPhases[state.backpack?.phase] || '识别画面'}${state.backpack?.capacity_after != null ? ` · ${state.backpack.capacity_after}/30` : ''}`;
  $('stage-text').textContent = !state.running ? `${activeCount} 项功能已开启，等待开始` : state.mode === 'preview' ? '仅识别画面，不发送点击' : state.stage === 'game_recovery' ? '游戏重连中 · 等待返回主画面' : state.stage.startsWith('stone_') ? `次元石：降阶刷最后BOSS · ${stonePairText()} · ${state.stone?.phase || '核对阶段'} · 第${state.stone?.world_stage ?? '—'}阶段 · RANK ${state.stone?.rank ?? '—'}` : state.stage.startsWith('meter_') ? `标尺等级校准 · ${state.meter?.pending ? '校准进行中' : (state.meter?.hold || '等待触发条件')}` : state.stage.startsWith('backpack_') ? bagText : state.stage.startsWith('skills_') ? `巡航系技能 · 钓鱼速度 ${state.skills?.fishing_percent ?? '—'}% · 技能点 ${state.skills?.balance ?? '待识别'}` : state.stage.startsWith('equipment_') ? `鱼饼武器强化 · ${state.equipment?.weapon || '识别列表'} · 本轮长按 ${state.equipment?.holds || 0} 次` : state.stage.startsWith('parallel_') ? (state.parallel?.phase === 'play' ? parallelText : `平行世界 · ${state.parallel?.phase || '识别中'} · RANK ${state.parallel?.rank ?? '—'}`) : state.stage === 'main' ? (state.parallel && state.features.find(f => f.id === 'parallel_world')?.enabled ? parallelText : state.active_rule_ids?.length ? '主画面 · 等待任务出现' : '所有功能已关闭，等待开启任务') : `当前流程 · ${stageRule?.name || state.stage}`;
  $('pending').hidden = !state.settings_pending;
  const health = state.lifecycle?.health;
  if (state.running && state.mode !== 'preview' && state.stage === 'main' && health) {
    const fresh = health.sample_age_seconds != null && health.sample_age_seconds <= 15;
    $('stage-text').textContent += ` · ${fresh ? health.reason : '等待新的实际钓鱼观测'}`;
  }
  if (state.running && state.lifecycle?.continuity?.pending_stage) $('stage-text').textContent += ' · 正在核对上次中断位置，保留原时限';
  $('stage-text').title = state.build?.id ? `构建 ${state.build.id} · 源码 ${state.build.source_sha256 || '开发版本'}` : '';
  for (const id of ['frames','matches','clicks']) $(id).textContent = state[id];
  const match = state.last_match;
  $('match').textContent = match ? `${match.name} · ${Math.round(match.score * 100)}%` : state.running ? '正在查找目标…' : '等待开始运行';
  $('device').textContent = `${state.serial} · ${state.resolution.join(' × ')}`;
  $('device-serial').textContent = state.serial; $('device-size').textContent = state.resolution.join(' × ');
  $('device-adb').textContent = state.settings.adb_path;
  if (!deviceAdbDirty) $('device-adb-path').value = state.settings.adb_path;
  if (!settingsDirty) { $('poll-interval').value = state.settings.poll_interval; $('max-clicks').value = state.settings.max_consecutive_clicks; }
  renderScheduler(); renderFeatures(); renderCustom(); renderEvents(); showNumbers(); route();
  if (state.last_error) notice(state.last_error);
  if (state.running && state.frame_id) {
    try { await loadFrame(state.frame_id); } catch (_) { $('capture-time').textContent = '画面更新失败，正在重试'; }
  }
  draw();
}

function draw() {
  if (!picture) return;
  ctx.drawImage(picture, 0, 0);
  const box = drag ? [Math.min(drag[0],drag[2]),Math.min(drag[1],drag[3]),Math.max(drag[0],drag[2]),Math.max(drag[1],drag[3])] : selection;
  if (box) { ctx.fillStyle = '#20a9dd25'; ctx.fillRect(box[0],box[1],box[2]-box[0],box[3]-box[1]); ctx.strokeStyle = '#42c6ff'; ctx.lineWidth = 3; ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]); }
  if (state?.last_match && state.running && currentId === state.frame_id) { const b = state.last_match.box; ctx.strokeStyle = '#43edab'; ctx.lineWidth = 4; ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]); }
}
async function loadFrame(id) {
  if (loadingImage || currentId === id || drag) return;
  loadingImage = true;
  try {
    const img = new Image();
    await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(Error('截图读取失败')); img.src = '/api/frame/' + id + '.png'; });
    picture = img; currentId = id; frame = {id,width:img.naturalWidth,height:img.naturalHeight}; selection = null;
    canvas.width = frame.width; canvas.height = frame.height; canvas.style.display = 'block'; $('empty').hidden = true;
    $('mini-image').src = img.src; $('mini-image').hidden = false; $('mini-empty').hidden = true;
    $('capture-time').textContent = `最近截图 ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`;
    $('selection').textContent = '停止运行后，可在画面上拖动框选新的识别目标。'; draw();
  } finally { loadingImage = false; }
}
function point(e) {
  const r = canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(canvas.width,Math.round((e.clientX-r.left)*canvas.width/r.width))),Math.max(0,Math.min(canvas.height,Math.round((e.clientY-r.top)*canvas.height/r.height)))];
}
canvas.addEventListener('pointerdown', e => { if (state?.running || busy || !frame) return; const p = point(e); drag = [...p,...p]; selection = null; canvas.setPointerCapture(e.pointerId); draw(); });
canvas.addEventListener('pointermove', e => { if (!drag) return; [drag[2],drag[3]] = point(e); draw(); });
canvas.addEventListener('pointerup', () => { if (!drag) return; selection = [Math.min(drag[0],drag[2]),Math.min(drag[1],drag[3]),Math.max(drag[0],drag[2]),Math.max(drag[1],drag[3])]; drag = null; $('selection').textContent = `已选区域：${selection.join(', ')} · ${selection[2]-selection[0]} × ${selection[3]-selection[1]}`; draw(); });
canvas.addEventListener('pointercancel', () => { drag = null; draw(); });

document.addEventListener('click', event => {
  const toggle = event.target.closest('[data-feature-switch]');
  if (toggle && !toggle.disabled) {
    event.preventDefault(); event.stopPropagation();
    const f = state.features.find(f => f.id === toggle.dataset.featureSwitch);
    act(() => api('/api/features/change', {id:f.id, enabled:!f.enabled}), `${f.name}已${f.enabled ? '关闭' : '开启'}并保存`);
  }
  const custom = event.target.closest('[data-custom-switch]');
  if (custom && !custom.disabled) act(() => api('/api/rules/change',{id:custom.dataset.customSwitch, enabled:custom.getAttribute('aria-checked') !== 'true'}), '目标开关已保存');
  const del = event.target.closest('[data-delete]');
  if (del && !del.disabled) act(() => api('/api/rules/change',{id:del.dataset.delete,delete:true}), '自定义目标已删除');
});
for (const id of ['log-scroll','log-scroll-full']) { const el = $(id); if (el) el.onchange = () => { for (const other of ['log-scroll','log-scroll-full']) { const peer = $(other); if (peer) peer.checked = el.checked; } }; }
$('run').onclick = () => act(() => api('/api/start',{preview:false}));
$('preview').onclick = () => act(() => api('/api/start',{preview:true}));
$('stop').onclick = () => act(() => api('/api/stop',{}));
$('refresh').onclick = $('refresh-mini').onclick = () => act(async () => { const f = await api('/api/capture'); await loadFrame(f.id); });
$('settings-form').addEventListener('input', () => { settingsDirty = true; });
$('settings-form').onsubmit = event => { event.preventDefault(); act(async () => { await api('/api/settings',{poll_interval:Number($('poll-interval').value),max_consecutive_clicks:Number($('max-clicks').value)}); settingsDirty = false; }, '运行设置已保存'); };
async function scanDevices() {
  if (deviceScanning || !connected || busy) return;
  deviceScanning = true;
  $('device-scan-state').textContent = '正在检测模拟器连接与画面尺寸…';
  updateControls();
  try {
    const result = await api('/api/devices/detect', {adb_path:$('device-adb-path').value});
    deviceScan = result.result;
    const previous = $('device-select').value;
    const options = deviceScan.devices.map(d => {
      const size = d.resolution ? ` · ${d.resolution.join('×')}` : '';
      return `<option value="${esc(d.serial)}" ${d.compatible ? '' : 'disabled'}>${esc(d.name)} · ${esc(d.serial)}${size} · ${esc(d.reason)}${d.selected ? '（当前）' : ''}</option>`;
    });
    $('device-select').innerHTML = '<option value="">请选择模拟器</option>' + options.join('');
    const chosen = deviceScan.devices.find(d => d.compatible && d.serial === previous)
      || deviceScan.devices.find(d => d.compatible && d.selected)
      || (deviceScan.automatic_available ? deviceScan.devices.find(d => d.compatible) : null);
    $('device-select').value = chosen?.serial || '';
    $('device-adb-path').value = deviceScan.adb_path;
    deviceAdbDirty = true;
    const count = deviceScan.devices.filter(d => d.compatible).length;
    $('device-scan-state').textContent = count ? `检测到 ${count} 个可连接实例。${count > 1 ? '请选择要连接的模拟器。' : '连接后仍保持停止，点击开始才运行。'}` : deviceScan.devices.length ? '检测到设备，但尚未就绪或尺寸不匹配；请查看列表中的说明。' : '没有检测到实例。请启动模拟器并开启 ADB 调试后重新检测。';
  } catch (error) {
    deviceScan = null;
    $('device-select').innerHTML = '<option value="">未检测到可连接实例</option>';
    $('device-scan-state').textContent = error.message;
  } finally {
    deviceScanning = false;
    updateControls();
  }
}
async function connectDevice(serial) {
  await act(async () => {
    const result = await api('/api/devices/select', {serial, adb_path:$('device-adb-path').value});
    generation++;
    frame = picture = selection = null;
    $('mini-image').hidden = true;
    $('mini-image').removeAttribute('src');
    $('mini-empty').hidden = false;
    deviceAdbDirty = false;
    await loadFrame(result.result.frame.id);
    $('device-scan-state').textContent = `已连接 ${result.result.name}，点击开始后执行任务。`;
    deviceScan = null;
    deviceScanStarted = false;
  }, '模拟器连接已保存');
}
$('device-detect').onclick = () => { deviceScanStarted = true; scanDevices(); };
$('device-adb-path').oninput = () => { deviceAdbDirty = true; deviceScan = null; updateControls(); };
$('device-select').onchange = updateControls;
$('device-auto').onclick = () => connectDevice('auto');
$('device-form').onsubmit = event => { event.preventDefault(); if ($('device-select').value) connectDevice($('device-select').value); };
$('target-form').onsubmit = event => {
  event.preventDefault();
  act(async () => {
    if (!frame || !selection) throw Error('请先在截图上拖动框选目标');
    await api('/api/rules',{frame_id:frame.id,area:selection,name:$('name').value,threshold:Number($('threshold').value),interval:Number($('interval').value),stable_frames:Number($('stable').value)});
    selection = null; $('name').value = ''; draw();
  }, '自定义目标已保存');
};
async function poll() {
  if (!busy) {
    try { await refreshStatus(); }
    catch (_) { connected = false; $('connection').dataset.state = 'off'; $('connection').innerHTML = '<i></i>已断开'; updateControls(); }
  }
  setTimeout(poll, 1200);
}
updateControls();
poll();
