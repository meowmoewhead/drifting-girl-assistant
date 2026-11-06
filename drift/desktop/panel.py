"""Native desktop UI. All task changes use the existing guarded runtime API."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk
from .client import Client, ensure_server, validate_address
from drift.update.package import checksum_for
from . import theme
from .theme import FONT, Card, Badge, Navigation, IconLabel, colors, text_colors

TZ = timezone(timedelta(hours=8))
CATEGORIES = ('干货', '炖煮', '生鱼片', '油炸', '炒饭', '面食', '寿司', '汤', '烧烤')
TRAINING_LABELS = dict(enabled='启用失败后练级', failure_limit='失败次数', level_gain='练级增量（兼容参数）',
    minutes='练级分钟数（兼容参数）', retry_level_gap='普通关允许等级差', final_boss_retry_level_gap='最后一关允许等级差',
    auto_sailing='练级自动航巡', meter_auto_adjust='练级标尺校准', boss_interrupt='海神反击', battle_tap='战斗补伤害',
    parallel_stage_limit='新门阶段上限（0不干预）', boss_auto_sailing_until_rank='早期 BOSS 航巡截止 RANK')


def execution_text(feature):
    execution = feature.get('execution') or {}
    if not feature.get('effective_enabled', feature.get('enabled')):
        return feature.get('inactive_reason') or '已关闭'
    if execution.get('status') == 'running':
        return '正在执行'
    if execution.get('mode') == 'realtime':
        return execution.get('condition') or '按当前画面实时检测'
    value = execution.get('next_run')
    if value:
        try:
            return '下次 ' + datetime.fromisoformat(value).astimezone(TZ).strftime('%m-%d %H:%M:%S')
        except ValueError:
            pass
    return execution.get('detail') or '等待条件满足'


class Panel:
    def __init__(self, window, root, client):
        self.window, self.root, self.client = window, Path(root), client
        self.state = {}
        self.page = 'overview'
        self.busy = False
        self.polling = False
        self.closed = False
        self.events = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix='desktop')
        self.fields = {}
        self.photo = None
        self.frame = None
        self.selection = None
        preference = theme.read_preferences(self.root).get('theme', 'system')
        self.theme_mode = preference if preference in theme.MODES.values() else 'system'
        self.theme_choice = tk.StringVar(value=next(k for k,v in theme.MODES.items() if v == self.theme_mode))
        self.window.title('漂流助手 · 桌面控制面板')
        self.window.geometry('1320x900')
        self.window.minsize(1040, 700)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.style()
        self.create_layout()
        self.window.after(40, self.drain)
        self.theme_timer = self.window.after(2500, self.watch_system_theme)
        self.poll()

    def style(self):
        theme.apply_theme(self.window, self.theme_mode)

    def change_theme(self, event=None):
        mode = theme.MODES.get(self.theme_choice.get(), self.theme_mode)
        try:
            theme.save_preferences(self.root, theme=mode)
        except (OSError, ValueError) as exc:
            self.theme_choice.set(next(k for k,v in theme.MODES.items() if v == self.theme_mode))
            self.notice.set('主题保存失败：' + str(exc))
            return
        self.theme_mode = mode
        self.style()
        self.notice_color()

    def watch_system_theme(self):
        if self.closed:
            return
        if self.theme_mode == 'system' and theme.resolve_theme('system') != self.window._drift_theme:
            self.style()
            self.notice_color()
        self.theme_timer = self.window.after(2500, self.watch_system_theme)

    def notice_color(self, *_):
        colors(self.notice_label, foreground='TEAL' if self.notice.get() in ('已完成。','正在处理…') else 'DANGER')

    def create_layout(self):
        sidebar = ttk.Frame(self.window, width=240, padding=(14, 25, 14, 14))
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        brand=ttk.Frame(sidebar)
        brand.pack(fill='x',padx=6,pady=(0,22))
        IconLabel(brand,'sail',38).pack(side='left',padx=(0,10))
        name=ttk.Frame(brand);name.pack(side='left')
        colors(ttk.Label(name,text='漂流助手',font=(FONT,17,'bold')),foreground='INK').pack(anchor='w')
        colors(ttk.Label(name,text='DRIFT ASSISTANT',font=('Segoe UI',7)),foreground='MUTED').pack(anchor='w',pady=(3,0))
        connection_box=colors(tk.Frame(sidebar,padx=12,pady=11),background='BG')
        connection_box.pack(fill='x',pady=(0,12))
        self.instance=tk.StringVar(value='正在连接模拟器')
        colors(tk.Label(connection_box,text='●',font=('Segoe UI',12)),background='BG',foreground='GREEN').pack(side='left',padx=(0,9))
        colors(tk.Label(connection_box,textvariable=self.instance,font=(FONT,9),anchor='w'),background='BG',foreground='INK').pack(fill='x')
        colors(tk.Label(connection_box,text='独立桌面控制面板',font=(FONT,8),anchor='w'),background='BG',foreground='MUTED').pack(fill='x',pady=(3,0))
        self.nav = Navigation(sidebar)
        self.nav.pack(fill='both', expand=True)
        self.nav.bind('<<TreeviewSelect>>', self.navigate)
        self.connection = tk.StringVar(value='正在连接…')
        ttk.Label(sidebar, textvariable=self.connection, wraplength=200, style='CardMuted.TLabel').pack(anchor='w', padx=9,pady=(15,0))
        colors(tk.Frame(self.window,width=1),background='BORDER').pack(side='left',fill='y')
        workspace = ttk.Frame(self.window,style='Page.TFrame')
        workspace.pack(side='left', fill='both', expand=True)
        bar = ttk.Frame(workspace,padding=(26,14,26,14))
        bar.pack(fill='x')
        self.breadcrumb=tk.StringVar(value='控制台  /  任务总览')
        colors(ttk.Label(bar,textvariable=self.breadcrumb,font=(FONT,9)),foreground='MUTED').pack(side='left')
        self.run_state = tk.StringVar(value='未连接')
        self.run_badge=Badge(bar,variable=self.run_state)
        self.run_badge.pack(side='left',padx=18)
        self.buttons = {}
        for label, path, data in [('停止', '/api/stop', {}),('开始', '/api/start', {'preview': False}),('识别预览', '/api/start', {'preview': True})]:
            button = ttk.Button(bar, text=label, style='Accent.TButton' if label == '开始' else 'Stop.TButton' if label=='停止' else 'TButton',
                                command=lambda p=path, d=data: self.post(p, d))
            button.pack(side='right', padx=4)
            self.buttons[label] = button
        self.theme_selector = ttk.Combobox(bar,textvariable=self.theme_choice,values=list(theme.MODES),state='readonly',width=9)
        self.theme_selector.pack(side='right',padx=(4,12))
        self.theme_selector.bind('<<ComboboxSelected>>',self.change_theme)
        colors(tk.Frame(workspace,height=1),background='BORDER').pack(fill='x')
        page=ttk.Frame(workspace,style='Page.TFrame',padding=(25,15,18,10))
        page.pack(fill='both',expand=True)
        self.notice = tk.StringVar()
        self.notice_label=ttk.Label(page,textvariable=self.notice,style='Muted.TLabel',wraplength=900)
        self.notice_label.pack(fill='x',pady=(0,4))
        self.notice.trace_add('write',self.notice_color)
        self.canvas = colors(tk.Canvas(page, highlightthickness=0),background='BG')
        scrollbar = ttk.Scrollbar(page, orient='vertical', command=self.canvas.yview)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = ttk.Frame(self.canvas,style='Page.TFrame',padding=(0,0,12,12))
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        self.content.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.content_id, width=e.width))
        self.window.bind('<MouseWheel>', self.scroll)
        ttk.Label(workspace,text='关闭窗口后保持后台运行  ·  需要停机时请使用“停止”',style='Muted.TLabel',font=(FONT,8)).pack(anchor='w',padx=26,pady=(2,12))
        self.render_page()

    def scroll(self, event):
        if str(event.widget).startswith(str(self.nav)):
            self.nav.canvas.yview_scroll(-int(event.delta/120),'units')
            return
        if isinstance(event.widget, (ttk.Treeview, tk.Text, ttk.Combobox)):
            return
        self.canvas.yview_scroll(-int(event.delta / 120), 'units')

    def close(self):
        self.closed = True
        self.window.after_cancel(self.theme_timer)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()

    def submit(self, function, callback=None, action=True):
        if action and self.busy:
            return
        if action:
            self.busy = True
            self.notice.set('正在处理…')
        def work():
            try:
                self.events.put((callback, function(), None, action))
            except Exception as exc:
                self.events.put((callback, None, str(exc), action))
        self.pool.submit(work)

    def drain(self):
        if self.closed:
            return
        while True:
            try:
                callback, result, error, action = self.events.get_nowait()
            except queue.Empty:
                break
            if action:
                self.busy = False
                self.notice.set(error or '已完成。')
            if callback:
                callback(result, error)
        self.window.after(60, self.drain)

    def post(self, path, data, callback=None):
        client = self.client
        self.submit(lambda: client.request(path, data), callback)

    def poll(self):
        if self.closed:
            return
        if not self.polling:
            self.polling = True
            client, page = self.client, self.page
            def get():
                if client.session is None:
                    client.connect()
                state = client.request('/api/status')
                update = client.request('/api/desktop/update') if page == 'update' else None
                return client, state, update
            def done(result, error):
                self.polling = False
                if error:
                    self.connection.set('未连接 · ' + error[:100])
                    self.run_state.set('连接不可用')
                    self.run_badge.set('连接不可用','red')
                    for button in self.buttons.values():
                        button.configure(state='disabled')
                    return
                source, state, update = result
                if source is not self.client:
                    return
                self.state = state
                self.connection.set(f'● 已连接   ·   v{state.get("version", "—")}\n配置自动保存到当前实例')
                self.instance.set('MuMu  ·  '+str(state.get('serial') or '当前实例'))
                self.run_state.set(('正在收尾' if state.get('mode') == 'stopping' else '运行中') if state.get('running') else '已停止')
                self.run_badge.set(self.run_state.get(),'green' if state.get('running') else 'gray')
                for label, button in self.buttons.items():
                    enabled = not self.busy and not state.get('config_missing') and (bool(state.get('running')) if label == '停止' else not state.get('running'))
                    button.configure(state='normal' if enabled else 'disabled')
                if not self.nav.get_children():
                    self.build_nav()
                    self.render_page()
                self.refresh_view(update)
            self.submit(get, done, action=False)
        self.window.after(1500, self.poll)

    def build_nav(self):
        self.nav.insert('', 'end', iid='overview', text='任务总览')
        features = {f['id']: f for f in self.state.get('features', [])}
        used = set()
        for group in self.state.get('task_groups', []):
            key = 'group/' + group['id']
            self.nav.insert('', 'end', iid=key, text=group['name'], open=False)
            for identifier in group['features']:
                if identifier in features:
                    self.nav.insert(key, 'end', iid='task/' + identifier, text=features[identifier]['name'])
                    used.add(identifier)
        for identifier, feature in features.items():
            if identifier not in used:
                self.nav.insert('', 'end', iid='task/' + identifier, text=feature['name'])
        for key, name in [('screen', '画面与识别'), ('logs', '运行日志'), ('settings', '运行与连接设置'), ('update', '检查更新')]:
            self.nav.insert('', 'end', iid=key, text=name)
        self.nav.selection_set(self.page)

    def navigate(self, event=None):
        selected = self.nav.selection()
        if not selected or selected[0] == self.page:
            return
        if self.fields and self.dirty() and not messagebox.askyesno('未保存的修改', '当前参数尚未保存，放弃这些修改并切换页面？', parent=self.window):
            self.nav.selection_set(self.page)
            return
        self.page = selected[0]
        self.render_page()

    def dirty(self):
        return any(variable.get() != initial for variable, initial, kind in self.fields.values())

    def title(self, text, subtitle=''):
        self.breadcrumb.set('控制台  /  '+text)
        ttk.Label(self.content, text=text, style='Title.TLabel').pack(anchor='w', pady=(4, 7))
        if subtitle:
            label=ttk.Label(self.content,text=subtitle,style='Muted.TLabel',wraplength=850)
            label.pack(anchor='w',pady=(0,16))
            self.content.bind('<Configure>',lambda e,w=label:w.configure(wraplength=max(300,e.width-35)) if w.winfo_exists() else None,add='+')

    def section(self, title):
        card=Card(self.content)
        card.pack(fill='x',pady=(0,15))
        ttk.Label(card.body,text=title,style='CardHeading.TLabel').pack(anchor='w',pady=(0,13))
        return card.body

    def field(self, parent, key, label, value, choices=None):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=4)
        if type(value) is bool:
            variable = tk.BooleanVar(value=value)
            ttk.Checkbutton(row, text=label, variable=variable).pack(anchor='w')
            kind = bool
        else:
            ttk.Label(row, text=label, width=32).pack(side='left')
            variable = tk.StringVar(value=str(value if value is not None else ''))
            widget = ttk.Combobox(row, textvariable=variable, values=choices, state='readonly', width=32) if choices else ttk.Entry(row, textvariable=variable, width=35)
            widget.pack(side='left', fill='x', expand=True)
            kind = type(value) if type(value) in (int, float) else str
        self.fields[key] = variable, variable.get(), kind
        return variable

    def check_grid(self,parent,entries,columns=3):
        grid=ttk.Frame(parent);grid.pack(fill='x',pady=(5,8))
        for column in range(columns):grid.columnconfigure(column,weight=1,uniform='checks')
        for i,(key,label,value) in enumerate(entries):
            cell=ttk.Frame(grid);cell.grid(row=i//columns,column=i%columns,sticky='w',padx=(0,12),pady=3)
            self.field(cell,key,label,value)

    def values(self):
        result = {}
        for key, (variable, _, kind) in self.fields.items():
            try:
                result[key] = kind(variable.get())
            except ValueError:
                raise ValueError('请检查数字参数，不要填写文字或空值') from None
        return result

    def safe(self, function):
        try:
            function()
        except Exception as exc:
            self.notice.set(str(exc))

    def render_page(self):
        # Drop callbacks owned by the previous page's wrapped subtitle.
        self.content.unbind('<Configure>')
        self.content.bind('<Configure>',lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        for widget in self.content.winfo_children():
            widget.destroy()
        self.fields = {}
        self.canvas.yview_moveto(0)
        self.photo = None
        self.frame = None
        self.selection = None
        self.log_box = None
        self.group_widgets={}
        self.task_rows={}
        self.overview_metrics={}
        self.overview_log=None
        self.page_icons=[]
        if self.page == 'overview' or self.page.startswith('group/'):
            self.render_overview()
        elif self.page.startswith('task/'):
            feature = next((f for f in self.state.get('features', []) if f['id'] == self.page[5:]), None)
            if feature:
                self.render_task(feature)
        elif self.page == 'settings':
            self.render_settings()
        elif self.page == 'screen':
            self.render_screen()
        elif self.page == 'update':
            self.render_update()
        elif self.page == 'logs':
            self.title('运行日志', '显示最近记录。完整日志保存在程序目录的 logs 文件夹。')
            box=self.section('近期记录')
            self.log_box = text_colors(tk.Text(box,height=29,wrap='word',font=(FONT,9),bd=0,padx=2,pady=4,spacing1=5,spacing3=5,highlightthickness=0))
            self.log_box.pack(fill='both', expand=True)
        self.refresh_view(None)

    def render_overview(self):
        group = next((g for g in self.state.get('task_groups', []) if self.page == 'group/' + g['id']), None)
        self.title(group['name'] if group else '任务总览',group['description'] if group else '安排任务，查看运行进度。实时任务按画面触发，定时任务按各自的时间执行。')
        if group:
            if group['id']=='stone':self.portal_controls(self.section('循环模式'))
            for feature in self.state.get('features',[]):
                if feature['id'] not in group['features']:continue
                card=Card(self.content,padding=18);card.pack(fill='x',pady=(0,11))
                row=ttk.Frame(card.body);row.pack(fill='x')
                text=ttk.Frame(row);text.pack(side='left',fill='both',expand=True)
                title=ttk.Frame(text);title.pack(anchor='w')
                ttk.Label(title,text=feature['name'],style='CardHeading.TLabel').pack(side='left',padx=(0,12))
                mode=(feature.get('execution') or {}).get('mode','realtime')
                Badge(title,{'daily':'每日','timed':'定时'}.get(mode,'实时'),{'daily':'gold','timed':'green'}.get(mode,'gray')).pack(side='left')
                detail=ttk.Label(text,text=feature.get('description',''),style='CardMuted.TLabel',wraplength=570)
                detail.pack(anchor='w',pady=(8,7))
                text.bind('<Configure>',lambda e,w=detail:w.configure(wraplength=max(280,e.width-16)))
                status=tk.StringVar();ttk.Label(text,textvariable=status,style='Status.TLabel',wraplength=570).pack(anchor='w')
                side=ttk.Frame(row);side.pack(side='right',padx=(18,0))
                badge=Badge(side);badge.pack(anchor='e',pady=(0,9))
                ttk.Button(side,text='设置  ›',command=lambda f=feature:self.open_page('task/'+f['id'])).pack(anchor='e')
                self.task_rows[feature['id']]=(status,badge)
            return
        metrics=ttk.Frame(self.content,style='Page.TFrame');metrics.pack(fill='x',pady=(0,18))
        for column,(key,label,caption) in enumerate([('enabled','已开启任务','独立开关 · 自动保存'),('frames','已读画面','本次脚本运行'),('clicks','执行输入','本次实际输入次数')]):
            metrics.columnconfigure(column,weight=1,uniform='metrics')
            card=Card(metrics,padding=18);card.grid(row=0,column=column,sticky='nsew',padx=(0,12 if column<2 else 0))
            ttk.Label(card.body,text=label,style='CardMuted.TLabel').pack(anchor='w')
            value=tk.StringVar(value='—');self.overview_metrics[key]=value
            ttk.Label(card.body,textvariable=value,style='Metric.TLabel').pack(anchor='w',pady=(5,3))
            ttk.Label(card.body,text=caption,style='CardMuted.TLabel').pack(anchor='w')
        log=Card(self.content,padding=17);log.pack(fill='x',pady=(0,20))
        heading=ttk.Frame(log.body);heading.pack(fill='x',pady=(0,9))
        ttk.Label(heading,text='运行日志',style='CardHeading.TLabel').pack(side='left')
        ttk.Button(heading,text='查看全部  ›',command=lambda:self.open_page('logs')).pack(side='right')
        self.overview_log=text_colors(tk.Text(log.body,height=3,wrap='none',font=(FONT,9),bd=0,highlightthickness=0,spacing1=5,spacing3=5,takefocus=False))
        self.overview_log.pack(fill='x');self.overview_log.tag_configure('time',foreground=theme.MUTED)
        head=ttk.Frame(self.content,style='Page.TFrame');head.pack(fill='x',pady=(0,12))
        ttk.Label(head,text='任务分组',style='Heading.TLabel').pack(side='left')
        ttk.Label(head,text='点击分组管理功能与执行时间',style='Muted.TLabel').pack(side='right')
        grid=ttk.Frame(self.content,style='Page.TFrame');grid.pack(fill='x')
        grid.columnconfigure(0,weight=1,uniform='groups');grid.columnconfigure(1,weight=1,uniform='groups')
        for i,g in enumerate(self.state.get('task_groups',[])):
            card=Card(grid,padding=18);card.grid(row=i//2,column=i%2,sticky='nsew',padx=(0,7) if i%2==0 else (7,0),pady=(0,14))
            top=ttk.Frame(card.body);top.pack(fill='x',pady=(0,12))
            IconLabel(top,g.get('icon','grid'),34).pack(side='left',padx=(0,10))
            ttk.Label(top,text=g['name'],style='CardHeading.TLabel').pack(side='left')
            badge=Badge(top);badge.pack(side='right')
            description=ttk.Label(card.body,text=g['description'],style='CardMuted.TLabel',wraplength=370)
            description.pack(fill='x',pady=(0,12))
            card.body.bind('<Configure>',lambda e,w=description:w.configure(wraplength=max(240,e.width-4)),add='+')
            bottom=ttk.Frame(card.body);bottom.pack(fill='x')
            ttk.Button(bottom,text='查看设置  ›',command=lambda key=g['id']:self.open_page('group/'+key)).pack(side='right')
            status=tk.StringVar()
            status_label=ttk.Label(bottom,textvariable=status,style='Status.TLabel',wraplength=145)
            status_label.pack(side='left',fill='x',expand=True,padx=(0,8))
            status_label.bind('<Configure>',lambda e,w=status_label:w.configure(wraplength=max(100,e.width)))
            self.group_widgets[g['id']]=(badge,status)

    def open_page(self,key):
        self.nav.selection_set(key)
        self.nav.see(key)

    def portal_controls(self, box):
        enabled = {f['id']: f.get('enabled') for f in self.state.get('features', [])}
        modes = {'降阶刷最后 BOSS': 'stone', '旧版任务导航练级': 'legacy', '关闭循环': 'off'}
        current = '降阶刷最后 BOSS' if enabled.get('stone_farm') else '旧版任务导航练级' if enabled.get('parallel_world') else '关闭循环'
        variable = tk.StringVar(value=current)
        ttk.Combobox(box, textvariable=variable, values=list(modes), state='readonly', width=32).pack(side='left')
        ttk.Button(box, text='保存模式（停止后）', command=lambda: self.post('/api/portal-mode', {'mode': modes[variable.get()]})).pack(side='left', padx=12)

    def render_task(self, feature):
        self.title(feature['name'], feature.get('description', ''))
        self.task_status = tk.StringVar()
        ttk.Label(self.content,textvariable=self.task_status,style='PageStatus.TLabel',wraplength=850).pack(anchor='w',pady=(0,15))
        box = self.section('功能开关')
        if feature['id'] in ('stone_farm', 'parallel_world'):
            self.portal_controls(box)
        else:
            variable = tk.BooleanVar(value=feature['enabled'])
            self.feature_enabled = variable
            ttk.Checkbutton(box, text='启用此功能（即时保存）', variable=variable,
                            command=lambda: self.post('/api/features/change', {'id': feature['id'], 'enabled': variable.get()})).pack(anchor='w')
        if feature['id'] != 'return_parallel':
            box = self.section('功能参数')
            self.field(box, 'interval', '检查 / 冷却间隔（秒）', float(feature.get('interval') or 1))
            if feature['id'] == 'mana':
                self.field(box, 'mana_threshold', '法力低于此值时补充', int(feature.get('mana_threshold') or 25))
            if feature.get('hold_seconds') is not None:
                ttk.Label(box, text=f'每种武器长按 {feature["hold_seconds"]} 秒，按当前配置执行。').pack(anchor='w')
            if feature.get('calibration'):
                c = feature['calibration']
                ttk.Label(box, text=f'差距严格超过 {c.get("trigger_gap")} 级；允许误差 ±{c.get("tolerance")} 级；失败退避 {c.get("failure_backoff")} 秒。', wraplength=800).pack(anchor='w')
            if feature.get('training'):
                for key, value in feature['training'].items():
                    self.field(box, 'training.' + key, TRAINING_LABELS.get(key, key), value)
            if feature['id'] == 'restaurant_upgrade':
                selected = set(feature.get('targets') or [])
                ttk.Label(box, text='选择要强化的类别；全部不选表示按图鉴顺序检查所有类别。').pack(anchor='w', pady=6)
                self.check_grid(box,[('category.'+c,c,c in selected) for c in CATEGORIES])
            if feature.get('cruise_settings'):
                options = feature['cruise_settings']
                cruise = self.section('游戏巡航')
                self.field(cruise, 'cruise.enabled', '管理巡航设置', options['cruise']['enabled'])
                ttk.Label(cruise,text='航巡目标',style='CardMuted.TLabel').pack(anchor='w',pady=(12,0))
                self.check_grid(cruise,[('cruise.targets.'+key,label,options['cruise']['targets'][key]) for key,label in [('pearl','珍珠'),('material','技能 / 天赋材料'),('amber','琥珀'),('gem','蓝色宝石'),('boss','BOSS')]])
                ttk.Separator(cruise).pack(fill='x',pady=(4,8))
                self.check_grid(cruise,[('cruise.'+key,label,options['cruise'][key]) for key,label in [('auto_sale','自动贩卖'),('auto_upgrade','自动强化')]])
                self.field(cruise,'cruise.ticket_min','贩卖券低于此数量时补充',options['cruise']['ticket_min'])
                skills = self.section('巡航系技能')
                self.field(skills, 'skills.enabled', '管理技能', options['skills']['enabled'])
                skill_grid=ttk.Frame(skills);skill_grid.pack(fill='x',pady=(10,15))
                for i,(key,label) in enumerate([('fishing','钓鱼加速'),('movement','移动加速'),('time','时间加速')]):
                    skill_grid.columnconfigure(i,weight=1,uniform='skills')
                    cell=ttk.Frame(skill_grid);cell.grid(row=0,column=i,sticky='nw')
                    ttk.Label(cell,text=label,style='CardHeading.TLabel').pack(anchor='w',pady=(0,7))
                    for field,tail in [('auto','自动续'),('enhance','技能强化')]:
                        self.field(cell,f'skills.{key}.{field}',tail,options['skills'][key][field])
                self.field(skills, 'skills.growth_points', '经验加速（仅使用技能点购买）', options['skills']['growth_points'])
            ttk.Button(self.content, text='保存功能参数', style='Accent.TButton', command=lambda: self.safe(lambda: self.save_task(feature))).pack(anchor='e', pady=10)
            ttk.Label(self.content, text='运行中可保存，沿用原热修改规则。当前流程的截止时间和已用次数不会重置。', style='Muted.TLabel', wraplength=850).pack(anchor='w')
        execution = feature.get('execution') or {}
        if execution.get('mode') != 'realtime':
            box = self.section('下次开启时间 · 北京时间')
            current = execution.get('next_run')
            try:
                value = datetime.fromisoformat(current).astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                value = datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')
            self.next_time = tk.StringVar(value=value)
            ttk.Entry(box, textvariable=self.next_time, width=28).pack(side='left')
            ttk.Button(box, text='保存时间', command=lambda: self.safe(lambda: self.save_schedule(feature['id']))).pack(side='left', padx=8)
            ttk.Button(box, text='设为可执行', command=lambda: self.post('/api/tasks/schedule', {'id': feature['id'], 'next_run': datetime.now(TZ).isoformat()})).pack(side='left')
        if feature['id'] in ('meter_calibration', 'stone_farm'):
            prefix = 'meter' if feature['id'] == 'meter_calibration' else 'stone'
            box = self.section('保存记录')
            self.record_text = tk.StringVar()
            ttk.Label(box, textvariable=self.record_text, wraplength=800).pack(anchor='w')
            ttk.Button(box, text='归档过期记录（停止后）', command=lambda: self.clear_record(prefix)).pack(anchor='w', pady=6)
        box = self.section('执行流程')
        for index, step in enumerate(feature.get('steps', []), 1):
            ttk.Label(box, text=f'{index}. {step}', wraplength=800).pack(anchor='w', pady=4)

    def save_task(self, feature):
        data = {'id': feature['id']}
        cruise = deepcopy(feature.get('cruise_settings'))
        training = deepcopy(feature.get('training'))
        targets = []
        for key, value in self.values().items():
            if key.startswith('category.'):
                if value:
                    targets.append(key.split('.', 1)[1])
            elif key.startswith('training.'):
                training[key.split('.', 1)[1]] = value
            elif key.startswith(('cruise.', 'skills.')):
                node = cruise
                parts = key.split('.')
                for part in parts[:-1]:
                    node = node[part]
                node[parts[-1]] = value
            else:
                data[key] = value
        if cruise is not None:
            data['cruise_settings'] = cruise
        if training is not None:
            data['training'] = training
        if feature['id'] == 'restaurant_upgrade':
            data['targets'] = targets
        self.post('/api/features/change', data, self.mark_saved)

    def mark_saved(self, result, error):
        if not error:
            self.fields = {k: (v, v.get(), kind) for k, (v, initial, kind) in self.fields.items()}

    def save_schedule(self, identifier):
        value = datetime.strptime(self.next_time.get().strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
        self.post('/api/tasks/schedule', {'id': identifier, 'next_run': value.isoformat()})

    def clear_record(self, prefix):
        if messagebox.askyesno('归档记录', '仅归档符合现有规则的过期记录，历史及已用次数保留。是否继续？', parent=self.window):
            record = self.state.get(prefix + '_record') or {}
            self.post('/api/' + prefix + '-clear', {'sha256': record.get('sha256')})

    def render_settings(self):
        self.title('运行与连接设置', '桌面窗口只连接本机服务。远程控制可填写已有的本机 SSH 转发地址。')
        box = self.section('外观')
        selector = ttk.Combobox(box,textvariable=self.theme_choice,values=list(theme.MODES),state='readonly',width=18)
        selector.pack(anchor='w')
        selector.bind('<<ComboboxSelected>>',self.change_theme)
        ttk.Label(box,text='立即生效并记住选择。跟随系统会自动适配 Windows 的应用颜色；切换保留当前页面和未保存参数。',
                  style='CardMuted.TLabel',wraplength=760).pack(anchor='w',pady=(9,0))
        box = self.section('面板连接')
        self.address = tk.StringVar(value=self.client.address)
        ttk.Entry(box, textvariable=self.address, width=43).pack(side='left', fill='x', expand=True)
        ttk.Button(box, text='连接', command=lambda: self.safe(self.change_connection)).pack(side='left', padx=8)
        box = self.section('运行参数')
        settings = self.state.get('settings') or {}
        self.field(box, 'poll_interval', '识别最小周期（秒）', float(settings.get('poll_interval') or .6))
        self.field(box, 'max_consecutive_clicks', '同一目标重复点击上限', int(settings.get('max_consecutive_clicks') or 20))
        ttk.Button(box, text='保存运行参数', command=lambda: self.safe(lambda: self.post('/api/settings', self.values(), self.mark_saved))).pack(anchor='e', pady=8)
        box = self.section('连接模拟器（先停止脚本）')
        self.adb_path = tk.StringVar(value=settings.get('adb_path', ''))
        ttk.Label(box, text='ADB 路径（留空自动查找）').pack(anchor='w')
        ttk.Entry(box, textvariable=self.adb_path).pack(fill='x', pady=5)
        ttk.Button(box, text='检测可连接的模拟器', command=self.detect_devices).pack(anchor='w')
        self.devices = []
        self.device_choice = ttk.Combobox(box, state='readonly')
        self.device_choice.pack(fill='x', pady=10)
        ttk.Button(box, text='连接所选模拟器', command=lambda: self.safe(self.select_device)).pack(anchor='e')
        ttk.Label(box, text=f'当前实例：{self.state.get("serial", "—")}    画面：{self.state.get("resolution", "—")}', style='CardMuted.TLabel').pack(anchor='w', pady=8)

    def change_connection(self):
        address = validate_address(self.address.get())
        client = Client(address)
        def done(result, error):
            if not error:
                self.client = client
                self.nav.delete(*self.nav.get_children())
                self.state = {}
                theme.save_preferences(self.root, address=address)
        self.submit(client.connect, done)

    def detect_devices(self):
        def done(result, error):
            if error:
                return
            scan = result['result']
            self.devices = [d for d in scan.get('devices', []) if d.get('compatible', False)]
            self.detected_adb = scan.get('adb_path', self.adb_path.get().strip())
            self.device_choice.configure(values=[d.get('label') or d.get('name') or d.get('serial') for d in self.devices])
            if self.devices:
                self.device_choice.current(0)
            self.notice.set(scan.get('message') or f'找到 {len(self.devices)} 个可用实例')
        self.post('/api/devices/detect', {'adb_path': self.adb_path.get().strip()}, done)

    def select_device(self):
        index = self.device_choice.current()
        if index < 0:
            raise ValueError('请先检测并选择模拟器')
        d = self.devices[index]
        self.post('/api/devices/select', {'serial': d['serial'], 'adb_path': self.detected_adb})

    def render_update(self):
        self.title('检查更新', '来源：meowmoewhead/drifting-girl-assistant 的 GitHub Releases。更新不会覆盖配置或任务记录。')
        box = self.section('版本与下载')
        self.update_message = tk.StringVar(value='正在读取更新状态…')
        ttk.Label(box, textvariable=self.update_message, wraplength=820).pack(anchor='w', pady=8)
        self.progress = ttk.Progressbar(box, maximum=100)
        self.progress.pack(fill='x', pady=10)
        actions = ttk.Frame(box)
        actions.pack(fill='x')
        self.update_buttons = {}
        for label, command in [('检查更新', lambda: self.post('/api/desktop/update/check', {})),
                               ('下载新版本', lambda: self.post('/api/desktop/update/download', {})),
                               ('安装并重启面板', self.install_update), ('导入离线包', lambda: self.safe(self.import_update))]:
            button = ttk.Button(actions, text=label, command=command)
            button.pack(side='left', padx=(0, 6))
            self.update_buttons[label] = button
        ttk.Label(box, text='安装前需停止脚本并等待收尾；更新后保持停止。下载失败不会影响正在运行的脚本。', style='CardMuted.TLabel', wraplength=820).pack(anchor='w', pady=12)
        settings = self.section('更新通道与代理')
        self.update_preview = tk.BooleanVar(value=False)
        self.update_proxy = tk.StringVar()
        self.update_options_loaded = False
        ttk.Checkbutton(settings, text='包含测试 / 预发布版本', variable=self.update_preview).pack(anchor='w')
        ttk.Label(settings, text='更新代理（可选，留空直连；例如 http://127.0.0.1:7890）').pack(anchor='w', pady=(8, 4))
        ttk.Entry(settings, textvariable=self.update_proxy).pack(fill='x')
        ttk.Button(settings, text='保存更新设置', command=lambda: self.post('/api/desktop/update/settings', {'preview': self.update_preview.get(), 'proxy': self.update_proxy.get()})).pack(anchor='e', pady=8)
        notes = self.section('更新说明')
        self.release_notes = text_colors(tk.Text(notes, height=14, wrap='word', font=(FONT, 10), relief='flat'))
        self.release_notes.pack(fill='both', expand=True)

    def install_update(self):
        if self.state.get('running'):
            self.notice.set('请先停止脚本，等待收尾完成。')
            return
        if not messagebox.askyesno('安装更新', '将备份旧程序并重启面板，配置和任务记录保留。更新完成后脚本保持停止。继续安装？', parent=self.window):
            return
        session = self.client.session or {}
        local = bool(getattr(sys, 'frozen', False)) and Path(session.get('executable', '')).resolve() == Path(sys.executable).resolve()
        data = {'client_pid': os.getpid()} if local else {}
        def done(result, error):
            if not error and result['result'].get('close_desktop'):
                self.close()
            elif not error:
                self.client.session = None
                self.notice.set('正在重启远端面板，窗口会自动重连。')
        self.post('/api/desktop/update/install', data, done)

    def import_update(self):
        path = filedialog.askopenfilename(title='选择 Windows 更新包', filetypes=[('Windows 发行包', '*.zip')], parent=self.window)
        if not path:
            return
        sums = Path(path).with_name('SHA256SUMS.txt')
        if not sums.is_file():
            selected = filedialog.askopenfilename(title='选择同版本的 SHA256SUMS.txt', filetypes=[('校验清单', '*.txt')], parent=self.window)
            if not selected:
                return
            sums = Path(selected)
        digest = checksum_for(sums.read_text('utf-8-sig'), Path(path).name)
        client = self.client
        self.submit(lambda: client.upload(path, digest))

    def render_screen(self):
        self.title('画面与识别', '只读取当前画面。停止脚本后可框选自定义目标，不会向游戏发送点击。')
        ttk.Button(self.content, text='刷新截图', command=self.capture).pack(anchor='w', pady=8)
        self.picture = colors(tk.Canvas(self.content, width=400, height=570, highlightthickness=0),background='SCREEN')
        self.picture.pack(anchor='w')
        self.picture.bind('<ButtonPress-1>', self.selection_start)
        self.picture.bind('<B1-Motion>', self.selection_move)
        self.picture.bind('<ButtonRelease-1>', self.selection_end)
        self.selection_text = tk.StringVar(value='点击刷新以读取新截图。')
        ttk.Label(self.content, textvariable=self.selection_text).pack(anchor='w', pady=7)
        box = self.section('自定义识别目标')
        self.field(box, 'name', '目标名称', '')
        self.field(box, 'threshold', '匹配阈值', .92)
        self.field(box, 'interval', '点击间隔（秒）', 1.)
        self.field(box, 'stable_frames', '连续确认帧数', 2)
        ttk.Button(box, text='保存框选目标', command=lambda: self.safe(self.save_selection)).pack(anchor='e')
        known = {r for f in self.state.get('features', []) for r in f.get('rules', [])}
        for rule in self.state.get('rules', []):
            if rule['id'] in known:
                continue
            row = ttk.Frame(box)
            row.pack(fill='x', pady=5)
            variable = tk.BooleanVar(value=rule['enabled'])
            ttk.Checkbutton(row, text=rule['name'], variable=variable, command=lambda r=rule, v=variable: self.post('/api/rules/change', {'id': r['id'], 'enabled': v.get()})).pack(side='left')
            ttk.Button(row, text='删除', command=lambda r=rule: self.delete_rule(r)).pack(side='right')

    def delete_rule(self, rule):
        if messagebox.askyesno('删除自定义目标', f'删除“{rule["name"]}”？', parent=self.window):
            self.post('/api/rules/change', {'id': rule['id'], 'delete': True})

    def capture(self):
        client = self.client
        def get():
            frame = client.request('/api/capture')
            return frame, client.request('/api/frame/' + frame['id'] + '.png', binary=True)
        def done(result, error):
            if error or self.page != 'screen':
                return
            frame, png = result
            image = Image.open(io.BytesIO(png))
            self.frame = frame
            self.scale = min(570 / image.height, 800 / image.width)
            self.photo = ImageTk.PhotoImage(image.resize((round(image.width * self.scale), round(image.height * self.scale)), Image.Resampling.LANCZOS))
            self.picture.configure(width=self.photo.width(), height=self.photo.height())
            self.picture.delete('all')
            self.picture.create_image(0, 0, image=self.photo, anchor='nw')
            self.selection = None
            self.selection_text.set('截图编号 ' + frame['id'] + ' · 可拖动框选')
        self.submit(get, done)

    def point(self, event):
        return [round(max(0, min(event.x / self.scale, self.frame['width']))), round(max(0, min(event.y / self.scale, self.frame['height'])))]

    def selection_start(self, event):
        if not self.frame or self.state.get('running'):
            return
        self.drag_start = self.point(event)

    def selection_move(self, event):
        if not getattr(self, 'drag_start', None) or self.state.get('running'):
            return
        a, b = self.drag_start, self.point(event)
        self.selection = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        self.picture.delete('selection')
        self.picture.create_rectangle(*[v * self.scale for v in self.selection], outline='#1c82ef', width=2, tags='selection')
        self.selection_text.set('已选区域：' + ', '.join(map(str, self.selection)))

    def selection_end(self, event):
        self.selection_move(event)
        self.drag_start = None

    def save_selection(self):
        if not self.frame or not self.selection or self.state.get('running'):
            raise ValueError('请先停止脚本，再刷新并框选目标')
        self.post('/api/rules', {**self.values(), 'frame_id': self.frame['id'], 'area': self.selection})

    def refresh_view(self, update):
        state = self.state
        if self.page == 'overview' or self.page.startswith('group/'):
            features=state.get('features',[])
            for key,value in self.overview_metrics.items():
                value.set(f'{sum(bool(f.get("effective_enabled",f.get("enabled"))) for f in features)} / {len(features)}' if key=='enabled' else f'{state.get(key,0):,}')
            for f in features:
                if f['id'] in self.task_rows:
                    status,badge=self.task_rows[f['id']]
                    status.set(execution_text(f))
                    on=f.get('effective_enabled',f.get('enabled'))
                    badge.set('已开启' if on else '已关闭','green' if on else 'gray')
            for group in state.get('task_groups',[]):
                if group['id'] not in self.group_widgets:continue
                badge,status=self.group_widgets[group['id']]
                items=[f for f in features if f['id'] in group['features']]
                active=[f for f in items if f.get('effective_enabled',f.get('enabled'))]
                badge.set(f'{len(active)} / {len(items)} 已开启','green' if active else 'gray')
                scheduled=sorted([f for f in active if (f.get('execution') or {}).get('next_run')],key=lambda f:f['execution']['next_run'])
                status.set(execution_text(scheduled[0]).split('\n')[0] if scheduled else '按画面条件自动执行' if active else '尚未开启任务')
            if self.overview_log is not None:
                self.overview_log.configure(state='normal');self.overview_log.delete('1.0','end')
                for entry in state.get('events',[])[-3:]:
                    self.overview_log.insert('end',entry.get('time','')+'   ','time')
                    self.overview_log.insert('end',entry.get('message','').replace('\n',' ')+'\n')
                if not state.get('events'):self.overview_log.insert('end','运行后将在这里显示近期记录。','time')
                self.overview_log.configure(state='disabled')
        elif self.page.startswith('task/') and hasattr(self, 'task_status'):
            feature = next((f for f in state.get('features', []) if f['id'] == self.page[5:]), None)
            if feature:
                if feature['id'] not in ('stone_farm', 'parallel_world') and hasattr(self, 'feature_enabled') and not self.busy:
                    self.feature_enabled.set(feature['enabled'])
                text = execution_text(feature)
                if state.get('settings_pending'):
                    text += '\n新设置已保存，等待当前操作结束后应用。'
                self.task_status.set(text)
                prefix = {'meter_calibration': 'meter', 'stone_farm': 'stone'}.get(feature['id'])
                if prefix and hasattr(self, 'record_text'):
                    record = state.get(prefix + '_record') or {}
                    self.record_text.set(record.get('error') or ('待处理：' + str(record.get('phase', '')) if record.get('pending') else '没有待处理的记录；历史与已用次数保留。'))
        elif self.page == 'logs' and self.log_box:
            lines = '\n'.join(f'{e.get("time", "")}  {e.get("message", "")}' for e in state.get('events', []))
            if self.log_box.get('1.0', 'end-1c') != lines:
                self.log_box.configure(state='normal')
                self.log_box.delete('1.0', 'end')
                self.log_box.insert('1.0', lines)
                self.log_box.see('end')
                self.log_box.configure(state='disabled')
        elif self.page == 'update' and update:
            self.update_message.set(f'当前版本 {update["current_version"]}\n' + (update.get('error') or update.get('message', '')))
            self.progress['value'] = update.get('progress', 0)
            if not self.update_options_loaded:
                self.update_preview.set(update['options']['preview'])
                self.update_proxy.set(update['options']['proxy'])
                self.update_options_loaded = True
            release = update.get('release') or {}
            notes = release.get('notes') or '检查更新后显示新版本说明。'
            last = update.get('last_install')
            if last:
                notes += '\n\n上次安装：' + (last.get('message') or last.get('error', ''))
            if self.release_notes.get('1.0', 'end-1c') != notes:
                self.release_notes.configure(state='normal')
                self.release_notes.delete('1.0', 'end')
                self.release_notes.insert('1.0', notes)
                self.release_notes.configure(state='disabled')
            for label, button in self.update_buttons.items():
                enabled = not update.get('busy') and not self.busy
                if label == '下载新版本':
                    enabled = enabled and bool(release)
                if label == '安装并重启面板':
                    enabled = enabled and update.get('installable') and not state.get('running')
                button.configure(state='normal' if enabled else 'disabled')
        if state.get('last_error') and not self.busy and not self.notice.get().startswith('正在'):
            self.notice.set('最近错误：' + str(state['last_error']))


def run(root, port=8765, address=None):
    root = Path(root).resolve()
    if os.name == 'nt':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            pass
    window = tk.Tk()
    if address is None and (root / 'desktop_settings.json').is_file():
        try:
            address = json.loads((root / 'desktop_settings.json').read_text('utf-8')).get('address')
        except (OSError, ValueError):
            pass
    client = Client(address or f'http://127.0.0.1:{port}')
    panel = Panel(window, root, client)
    # Start only the local service, only on a refused connection; never start tasks.
    if address is None or address == f'http://127.0.0.1:{port}':
        def done(result, error):
            if not error:
                panel.client = result
        panel.submit(lambda: ensure_server(root, port), done)
    window.mainloop()
