"""Live light/dark palettes for the native console; no runtime settings change."""
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image, ImageDraw, ImageTk
from drift.update.package import atomic_json

FONT = 'Microsoft YaHei UI'
MODES = {'跟随系统': 'system', '浅色': 'light', '深色': 'dark'}
PALETTES = {
    'light': dict(BG='#f5f7f9', WHITE='#ffffff', INK='#243e4e', TEXT='#536b7a',
        MUTED='#61737f', BORDER='#e0e7ec', TEAL='#217d70', GREEN='#2e9567',
        SOFT='#eaf4f0', HOVER='#f0f5f4', FIELD='#ffffff', FOCUS='#549f93',
        SCROLL='#c4d0d8', SCROLL_HOVER='#9aafbc', TRACK='#e9eff2',
        DISABLED='#7b8c96', DISABLED_BG='#eef2f5', ACCENT_BG='#217d70',
        ACCENT_HOVER='#19695e', ON_ACCENT='#ffffff', SWITCH_OFF='#c3d0d8',
        SWITCH_DISABLED='#a9c3b7', KNOB='#ffffff', DANGER='#aa473e',
        DANGER_BG='#fdf0ed', DANGER_HOVER='#f8e3dd', DANGER_BORDER='#edcec9',
        STATUS='#486f64', SCREEN='#e2e7ef', SELECT_BG='#cce8e0', SELECT_FG='#173e35',
        BADGES={'green': ('#eaf4f0', '#27795a'), 'gray': ('#eef2f5', '#5d6f7c'),
                'gold': ('#f8f1df', '#866a2b'), 'red': ('#fdf0ed', '#aa473e'),
                'teal': ('#eaf3f5', '#456f82')}),
    'dark': dict(BG='#11191f', WHITE='#1b262f', INK='#e3edf3', TEXT='#bdcdd7',
        MUTED='#97aab8', BORDER='#344650', TEAL='#70cdbb', GREEN='#51b98b',
        SOFT='#203c37', HOVER='#263640', FIELD='#15212a', FOCUS='#70cdbb',
        SCROLL='#3f5665', SCROLL_HOVER='#668294', TRACK='#111c24',
        DISABLED='#788e9e', DISABLED_BG='#24333e', ACCENT_BG='#246f65',
        ACCENT_HOVER='#2d8479', ON_ACCENT='#ffffff', SWITCH_OFF='#4a6272',
        SWITCH_DISABLED='#3e6859', KNOB='#e7f1f5', DANGER='#f2aaa3',
        DANGER_BG='#432e31', DANGER_HOVER='#56363a', DANGER_BORDER='#68454a',
        STATUS='#91c8b3', SCREEN='#101920', SELECT_BG='#315e56', SELECT_FG='#effbf7',
        BADGES={'green': ('#203c32', '#91d9b8'), 'gray': ('#2a3741', '#acbcc8'),
                'gold': ('#42381f', '#e5cb8c'), 'red': ('#452c2d', '#f2a7a2'),
                'teal': ('#213944', '#98cfdf')})}
globals().update(PALETTES['light'])


def read_preferences(root):
    try:
        value = json.loads((Path(root) / 'desktop_settings.json').read_text('utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_preferences(root, **changes):
    """Merge only local desktop preferences; never write the game config."""
    path = Path(root) / 'desktop_settings.json'
    if path.exists():
        try:
            value = json.loads(path.read_text('utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError('桌面设置格式不正确，未覆盖原文件')
        except (OSError, ValueError) as exc:
            raise ValueError('桌面设置无法读取，未覆盖原文件') from exc
    else:
        value = {}
    value.update(changes)
    atomic_json(path, value)


def system_theme():
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as key:
                return 'light' if winreg.QueryValueEx(key, 'AppsUseLightTheme')[0] else 'dark'
        except OSError:
            pass
    return 'light'


def resolve_theme(mode):
    return mode if mode in PALETTES else system_theme()


def colors(widget, **tokens):
    """Retain semantic colors so existing widgets recolor without losing edits."""
    widget._drift_colors = {**getattr(widget, '_drift_colors', {}), **tokens}
    widget.configure(**{option: globals()[token] for option, token in widget._drift_colors.items()})
    return widget


def text_colors(widget):
    return colors(widget, background='WHITE', foreground='TEXT', insertbackground='INK',
                  selectbackground='SELECT_BG', selectforeground='SELECT_FG')


def _titlebar(window, dark):
    if os.name != 'nt':
        return
    try:
        import ctypes
        from ctypes import wintypes
        window.update_idletasks()
        user = ctypes.windll.user32
        user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user.GetAncestor.restype = wintypes.HWND
        hwnd = user.GetAncestor(window.winfo_id(), 2)
        enabled = wintypes.BOOL(dark)
        setter = ctypes.windll.dwmapi.DwmSetWindowAttribute
        setter.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        setter(hwnd, 20, ctypes.byref(enabled), ctypes.sizeof(enabled))
    except (OSError, AttributeError, tk.TclError):
        pass


def rounded(canvas, x1, y1, x2, y2, radius=9, **options):
    r = min(radius, (x2-x1)/2, (y2-y1)/2)
    return canvas.create_polygon(x1+r,y1,x2-r,y1,x2,y1,x2,y1+r,
        x2,y2-r,x2,y2,x2-r,y2,x1+r,y2,x1,y2,x1,y2-r,x1,y1+r,x1,y1,
        smooth=True, splinesteps=24, **options)


class Card(tk.Canvas):
    def __init__(self, parent, padding=20, **kwargs):
        super().__init__(parent, bg=BG, highlightthickness=0, bd=0, height=80, **kwargs)
        self.padding = padding
        self.body = ttk.Frame(self, style='Card.TFrame')
        self.slot = self.create_window(padding, padding, window=self.body, anchor='nw')
        self.bind('<Configure>', self.layout)
        self.body.bind('<Configure>', self.fit)

    def refresh_theme(self):
        self.configure(bg=BG)
        self.layout()

    def fit(self, event=None):
        height = self.body.winfo_reqheight() + 2*self.padding
        if int(self.cget('height')) != height:
            self.configure(height=height)

    def layout(self, event=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete('surface')
        rounded(self, 1, 1, w-1, h-1, fill=WHITE, outline=BORDER, tags='surface')
        self.tag_lower('surface')
        self.itemconfigure(self.slot, width=max(1,w-2*self.padding))


class Badge(tk.Canvas):
    def __init__(self, parent, text='', tone='gray', variable=None, **kwargs):
        super().__init__(parent, bg=WHITE, highlightthickness=0, height=24, **kwargs)
        self.face = tkfont.Font(family=FONT, size=9)
        self.tone = tone
        self.set(text, tone)
        self.variable, self.trace = variable, None
        if variable is not None:
            self.trace = variable.trace_add('write', lambda *_: self.set(variable.get(), self.tone))
            self.set(variable.get(), tone)
            self.bind('<Destroy>', self.cleanup)

    def cleanup(self, event):
        if event.widget is self and self.trace:
            self.variable.trace_remove('write', self.trace)
            self.trace = None

    def set(self, text, tone=None):
        self.text = text
        self.tone = tone or self.tone
        bg, fg = BADGES[self.tone]
        width = max(48,self.face.measure(str(text))+20)
        self.configure(width=width)
        self.delete('all')
        rounded(self,0,0,width,24,11,fill=bg,outline='')
        self.create_text(width/2,12,text=text,font=self.face,fill=fg)

    def refresh_theme(self):
        self.configure(bg=WHITE)
        self.set(self.text)


def icon(master, name, size=34, color=None, background=None):
    color, background = color or TEAL, background or SOFT
    scale=3
    image=Image.new('RGBA',(size*scale,size*scale),(0,0,0,0))
    d=ImageDraw.Draw(image)
    d.rounded_rectangle((0,0,size*scale-1,size*scale-1),radius=9*scale,fill=background)
    def points(values):return [(x*size*scale/32,y*size*scale/32) for x,y in values]
    def line(values):d.line(points(values),fill=color,width=max(2,round(size*scale/18)),joint='curve')
    def rect(box):d.rounded_rectangle(tuple(v*size*scale/32 for v in box),radius=2*scale,outline=color,width=max(2,round(size*scale/18)))
    def ellipse(box):d.ellipse(tuple(v*size*scale/32 for v in box),outline=color,width=max(2,round(size*scale/18)))
    if name=='sail':
        line([(16,6),(16,22),(8,22),(11,26),(23,26),(26,22),(16,22)])
        line([(14,8),(7,19),(14,19)]);line([(19,10),(25,19),(19,19)])
    elif name=='portal':
        ellipse((6,6,26,26));line([(16,10),(21,16),(16,22),(11,16),(16,10)])
    elif name=='growth':
        line([(15,26),(15,15),(21,9)]);line([(16,17),(10,16),(7,10),(14,11),(16,17)])
        line([(17,15),(24,13),(26,6),(19,8),(17,15)])
    elif name=='gift':
        rect((6,13,26,18));rect((8,18,24,27));line([(16,13),(16,26)])
        line([(16,12),(13,6),(9,6),(8,9),(11,12),(23,12),(25,9),(23,6),(20,6),(16,12)])
    elif name=='coin':
        ellipse((6,6,26,26));ellipse((10,10,22,22));line([(15,12),(15,20),(19,20)])
    elif name=='flask':
        line([(12,6),(20,6)]);line([(13,6),(13,13),(6,25),(8,27),(24,27),(26,25),(19,13),(19,6)])
        line([(10,20),(22,20)])
    elif name=='swords':
        line([(7,6),(24,24),(26,22)]);line([(25,6),(8,24),(6,22)])
        line([(18,23),(23,18)]);line([(9,18),(14,23)])
    elif name=='screen':
        rect((5,7,27,23));line([(16,23),(16,27),(11,27),(21,27)])
    elif name=='logs':
        rect((8,5,24,27));line([(12,11),(20,11)]);line([(12,16),(20,16)]);line([(12,21),(18,21)])
    elif name=='settings':
        for x,y in [(9,12),(16,21),(23,11)]:
            line([(x,6),(x,26)]);rect((x-3,y-2,x+3,y+2))
    elif name=='update':
        line([(9,12),(9,9),(16,6),(23,9),(26,15)]);line([(26,8),(26,15),(19,15)])
        line([(23,20),(23,23),(16,26),(9,23),(6,17)]);line([(6,24),(6,17),(13,17)])
    else:
        for x,y in [(7,7),(18,7),(7,18),(18,18)]:rect((x,y,x+7,y+7))
    return ImageTk.PhotoImage(image.resize((size,size),Image.Resampling.LANCZOS),master=master)


class IconLabel(ttk.Label):
    def __init__(self, parent, name, size=34, **kwargs):
        super().__init__(parent, **kwargs)
        self.icon_name, self.icon_size = name, size
        self.refresh_theme()

    def refresh_theme(self):
        self.photo = icon(self, self.icon_name, self.icon_size)
        self.configure(image=self.photo)


def apply_theme(window, mode='light'):
    name = resolve_theme(mode)
    globals().update(PALETTES[name])
    window._drift_theme = name
    style=ttk.Style(window);style.theme_use('clam')
    window.configure(bg=BG)
    style.configure('.',font=(FONT,10),background=WHITE,foreground=TEXT)
    style.configure('TFrame',background=WHITE)
    style.configure('Card.TFrame',background=WHITE)
    style.configure('Page.TFrame',background=BG)
    style.configure('TLabel',background=WHITE,foreground=TEXT)
    style.configure('Title.TLabel',background=BG,foreground=INK,font=(FONT,20,'bold'))
    style.configure('Heading.TLabel',background=BG,foreground=INK,font=(FONT,11,'bold'))
    style.configure('CardHeading.TLabel',background=WHITE,foreground=INK,font=(FONT,11,'bold'))
    style.configure('Muted.TLabel',background=BG,foreground=MUTED,font=(FONT,9))
    style.configure('CardMuted.TLabel',background=WHITE,foreground=MUTED,font=(FONT,9))
    style.configure('Metric.TLabel',background=WHITE,foreground=INK,font=(FONT,21,'bold'))
    style.configure('Status.TLabel',background=WHITE,foreground=STATUS,font=(FONT,9))
    style.configure('PageStatus.TLabel',background=BG,foreground=STATUS,font=(FONT,9))
    style.configure('TEntry',padding=(10,8),fieldbackground=FIELD,bordercolor=BORDER,lightcolor=FIELD,darkcolor=FIELD,insertcolor=INK,selectbackground=SELECT_BG,selectforeground=SELECT_FG)
    style.map('TEntry',bordercolor=[('focus',FOCUS)],foreground=[('disabled',DISABLED)],fieldbackground=[('disabled',DISABLED_BG),('readonly',FIELD)])
    style.configure('TCombobox',padding=(10,7),fieldbackground=FIELD,background=WHITE,bordercolor=BORDER,arrowcolor=MUTED,selectbackground=SELECT_BG,selectforeground=SELECT_FG)
    style.map('TCombobox',fieldbackground=[('disabled',DISABLED_BG),('readonly',FIELD)],foreground=[('disabled',DISABLED),('readonly',TEXT)],selectbackground=[('readonly',FIELD)],selectforeground=[('readonly',TEXT)],bordercolor=[('focus',FOCUS)],background=[('active',HOVER),('pressed',HOVER)],arrowcolor=[('disabled',DISABLED),('active',TEAL)])
    style.configure('Vertical.TScrollbar',arrowsize=8,gripcount=0,background=SCROLL,bordercolor=BG,troughcolor=BG,lightcolor=BG,darkcolor=BG)
    style.layout('Vertical.TScrollbar',[('Vertical.Scrollbar.trough',{'sticky':'ns','children':[('Vertical.Scrollbar.thumb',{'expand':'1','sticky':'nswe'})]})])
    style.map('Vertical.TScrollbar',background=[('active',SCROLL_HOVER)])
    style.configure('Horizontal.TProgressbar',background=TEAL,troughcolor=TRACK,bordercolor=TRACK,lightcolor=TEAL,darkcolor=TEAL)
    assets = getattr(window, '_theme_assets', {})
    images = assets.get(name, [])
    # Element images are created once per palette/Tk interpreter and retained.
    # Recreating/deleting their PhotoImages would blank buttons on a second switch.
    for key,normal,hover,foreground,border in [
        ('TButton',WHITE,HOVER,TEXT,BORDER),
        ('Accent.TButton',ACCENT_BG,ACCENT_HOVER,ON_ACCENT,ACCENT_BG),
        ('Stop.TButton',DANGER_BG,DANGER_HOVER,DANGER,DANGER_BORDER)]:
        element='Drift.'+name+'.'+key
        if element not in style.element_names():
            states=[]
            for bg,edge in [(normal,border),(hover,border),(DISABLED_BG,BORDER),(normal,FOCUS)]:
                im=Image.new('RGBA',(72,72),(0,0,0,0));draw=ImageDraw.Draw(im)
                draw.rounded_rectangle((1,1,70,70),radius=17,fill=bg,outline=edge,width=2)
                photo=ImageTk.PhotoImage(im.resize((36,36),Image.Resampling.LANCZOS),master=window)
                images.append(photo);states.append(photo)
            style.element_create(element,'image',states[0],('disabled',states[2]),('pressed',states[1]),('active',states[1]),('focus',states[3]),border=9,sticky='nswe')
        style.layout(key,[(element,{'sticky':'nswe','children':[('Button.padding',{'sticky':'nswe','children':[('Button.label',{'sticky':'nswe'})]})]})])
        style.configure(key,padding=(10,3),foreground=foreground,font=(FONT,9),anchor='center',background=WHITE)
        style.map(key,foreground=[('disabled',DISABLED)],background=[('disabled',WHITE),('active',WHITE),('pressed',WHITE)])
    switch_element='Drift.'+name+'.switch'
    if switch_element not in style.element_names():
        switches=[]
        for selected,disabled in [(False,False),(True,False),(False,True),(True,True)]:
            im=Image.new('RGBA',(120,66),(0,0,0,0));draw=ImageDraw.Draw(im)
            fill=(SWITCH_DISABLED if selected else BORDER) if disabled else (GREEN if selected else SWITCH_OFF)
            draw.rounded_rectangle((0,3,102,63),radius=30,fill=fill)
            x=48 if selected else 6;draw.ellipse((x,9,x+48,57),fill=KNOB)
            photo=ImageTk.PhotoImage(im.resize((40,22),Image.Resampling.LANCZOS),master=window)
            images.append(photo);switches.append(photo)
        style.element_create(switch_element,'image',switches[0],('disabled','selected',switches[3]),('disabled',switches[2]),('selected',switches[1]),width=44,sticky='w')
    style.layout('TCheckbutton',[('Checkbutton.padding',{'sticky':'nswe','children':[(switch_element,{'side':'left','sticky':'w'}),('Checkbutton.focus',{'side':'left','sticky':'w','children':[('Checkbutton.label',{'sticky':'nswe'})]})]})])
    style.configure('TCheckbutton',padding=(0,5),background=WHITE,foreground=TEXT)
    style.map('TCheckbutton',background=[('active',WHITE)],foreground=[('disabled',DISABLED)])
    assets[name] = images
    window._theme_assets = assets
    for option, value in dict(background=FIELD,foreground=TEXT,selectBackground=SELECT_BG,selectForeground=SELECT_FG).items():
        window.option_add('*TCombobox*Listbox.'+option,value)

    def refresh(widget):
        if getattr(widget, '_drift_colors', None):
            colors(widget)
        callback = getattr(widget, 'refresh_theme', None)
        if callable(callback):
            callback()
        if isinstance(widget, tk.Text) and 'time' in widget.tag_names():
            widget.tag_configure('time', foreground=MUTED)
        if isinstance(widget, ttk.Combobox):
            # Tcl creates the list only when first opened; update existing lists too.
            popdown = str(widget)+'.popdown.f.l'
            if window.tk.call('winfo','exists',popdown):
                window.tk.call(popdown,'configure','-background',FIELD,'-foreground',TEXT,
                               '-selectbackground',SELECT_BG,'-selectforeground',SELECT_FG)
        for child in widget.winfo_children():
            refresh(child)
    refresh(window)
    _titlebar(window, name == 'dark')
    return name


class Navigation(tk.Frame):
    """Keyboard-accessible sidebar; small tree-compatible adapter for page routing."""
    def __init__(self,parent):
        super().__init__(parent,bg=WHITE)
        self.records={};self.selected=();self.rows={};self.icons={};self.pending=None
        self.canvas=tk.Canvas(self,bg=WHITE,bd=0,highlightthickness=0)
        self.canvas.pack(side='left',fill='both',expand=True)
        self.body=tk.Frame(self.canvas,bg=WHITE)
        self.slot=self.canvas.create_window(0,0,window=self.body,anchor='nw')
        self.body.bind('<Configure>',lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',lambda e:self.canvas.itemconfigure(self.slot,width=e.width))

    def refresh_theme(self):
        self.configure(bg=WHITE)
        self.canvas.configure(bg=WHITE)
        self.body.configure(bg=WHITE)
        self.icons.clear()
        self.redraw_later()

    def insert(self,parent,index,iid,text,open=False):
        self.records[iid]=dict(parent=parent,text=text,open=open)
        self.redraw_later()

    def get_children(self,parent=''):
        return tuple(k for k,v in self.records.items() if v['parent']==parent)

    def delete(self,*keys):
        for key in keys:
            self.delete(*self.get_children(key));self.records.pop(key,None)
        self.redraw_later()

    def selection(self):return self.selected

    def selection_set(self,key):
        if key not in self.records:return
        self.selected=(key,)
        parent=self.records[key]['parent']
        if parent:self.records[parent]['open']=True
        self.redraw_later();self.event_generate('<<TreeviewSelect>>')

    def see(self,key):
        def reveal():
            if key not in self.rows:return
            self.update_idletasks();row=self.rows[key];h=max(1,self.body.winfo_height())
            y=row.winfo_y();first,last=self.canvas.yview()
            if y<first*h:self.canvas.yview_moveto(y/h)
            elif y+row.winfo_height()>last*h:self.canvas.yview_moveto((y+row.winfo_height()-self.canvas.winfo_height())/h)
        self.after_idle(reveal)

    def redraw_later(self):
        if self.pending is None:self.pending=self.after_idle(self.redraw)

    def select(self,key):
        if self.get_children(key):self.records[key]['open']=not self.records[key]['open']
        self.selection_set(key)

    def key_move(self,key,delta):
        keys=list(self.rows);i=max(0,min(len(keys)-1,keys.index(key)+delta));self.selection_set(keys[i])
        self.after_idle(lambda:self.rows.get(keys[i],self.canvas).focus_set());self.see(keys[i])

    def redraw(self):
        self.pending=None
        focused=self.focus_get();focus_key=next((k for k,v in self.rows.items() if v is focused),None)
        for widget in self.body.winfo_children():widget.destroy()
        self.rows={}
        for key,r in self.records.items():
            if r['parent'] and not self.records[r['parent']]['open']:continue
            if key in ('group/cruise','screen'):
                bar=tk.Frame(self.body,bg=WHITE);bar.pack(fill='x',pady=(13,6))
                tk.Label(bar,text='任务管理' if key.startswith('group') else '工作台',fg=MUTED,bg=WHITE,font=(FONT,8)).pack(anchor='w',padx=12)
            row=tk.Canvas(self.body,height=39 if not r['parent'] else 35,bg=WHITE,bd=0,highlightthickness=0,takefocus=True,cursor='hand2')
            row.pack(fill='x',pady=1);self.rows[key]=row
            def paint(event=None,k=key,w=row,record=r,hover=False):
                w.delete('all');selected=self.selected==(k,);width=w.winfo_width();height=w.winfo_height()
                if selected or hover:rounded(w,1,1,width-2,height-1,7,fill=SOFT if selected else HOVER,outline='')
                color=TEAL if selected else TEXT
                child=bool(record['parent']);x=25 if child else 12
                if child:w.create_oval(15,15,18,18,fill=GREEN if selected else SCROLL,outline='')
                if not child:
                    symbol={'overview':'grid','screen':'screen','logs':'logs','settings':'settings','update':'update',
                        'group/cruise':'sail','group/stone':'portal','group/restaurant':'growth','group/rewards':'gift',
                        'group/guild':'coin','group/activities':'flask','group/raid':'swords','group/recovery':'screen'}.get(k,'grid')
                    cache=(symbol,selected,hover)
                    if cache not in self.icons:self.icons[cache]=icon(self,symbol,21,color,SOFT if selected else HOVER if hover else WHITE)
                    w.create_image(20,height/2,image=self.icons[cache]);x=39
                w.create_text(x,height/2,text=record['text'],anchor='w',fill=color,font=(FONT,9,'bold' if selected else 'normal'))
                if self.get_children(k):w.create_text(width-14,height/2,text='⌄' if record['open'] else '›',fill=MUTED,font=('Segoe UI',13))
                if w is self.focus_get():rounded(w,1,1,width-2,height-1,7,fill='',outline=FOCUS)
            row.bind('<Configure>',paint);row.bind('<Enter>',lambda e,p=paint:p(hover=True));row.bind('<Leave>',paint)
            row.bind('<FocusIn>',paint);row.bind('<FocusOut>',paint)
            row.bind('<Button-1>',lambda e,k=key:self.select(k))
            row.bind('<Return>',lambda e,k=key:self.select(k));row.bind('<space>',lambda e,k=key:self.select(k))
            row.bind('<Up>',lambda e,k=key:self.key_move(k,-1));row.bind('<Down>',lambda e,k=key:self.key_move(k,1))
        if focus_key in self.rows:self.rows[focus_key].focus_set()
