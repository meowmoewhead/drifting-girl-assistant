"""页面图:把配置里的 ``stage → next_stage`` 规则当作导航边。

ALAS 的 ``module/ui/page.py`` 手写 ``Page.link`` 再反向 BFS 求 parent;本项目的
转移关系本来就在规则里(每条点击规则的 ``stage`` 是所在页,``next_stage`` 是目标页),
所以这里直接从规则推导同一张图,不需要额外的素材或手写连线。

导航只做一件事:给当前页算出"朝目标页走一步"该点哪条规则。
不发输入、不看截图,由调用方(runtime)决定何时点、怎么点。
"""

from __future__ import annotations

MAIN = 'main'


def action(rule):
    """规则的 action:配置里 ``None``(键存在但为空)表示普通点击。"""
    return rule.get('action') or 'click'


# 导航只允许"退回"性质的动作。购买/数量/确认类规则会把状态往前推(花资源、
# 改库存),绝不能当成回主界面的路 —— 宁可报 no_route,也不能顺手买东西。
_PURCHASE_KEYS = ('purchase_condition', 'quantity_condition', 'confirmation_condition')
_PURCHASE_HINTS = ('_buy', '_plus')


def navigable(rule):
    """这条规则能不能作为导航边。"""
    if any(rule.get(key) for key in _PURCHASE_KEYS):
        return False
    rid = rule.get('id', '')
    return not any(hint in rid for hint in _PURCHASE_HINTS)


class Page:
    """一个页面:用一组"只可能出现在这一页"的规则 id 作为检测按钮。"""

    all_pages: dict[str, 'Page'] = {}

    def __init__(self, name, check=()):
        self.name = name
        self.check = tuple(check)
        self.links: dict[str, str] = {}  # 目标页名 → 点击该边的规则 id
        self.parent = None
        Page.all_pages[name] = self

    def link(self, button, destination):
        """登记一条边:在点击 ``button`` 后到达 ``destination``。"""
        self.links[destination] = button
        return self

    def __eq__(self, other):
        return isinstance(other, Page) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Page({self.name})'

    @classmethod
    def clear(cls):
        cls.all_pages.clear()

    @classmethod
    def iter_pages(cls):
        return list(cls.all_pages.values())

    @classmethod
    def clear_connection(cls):
        for page in cls.all_pages.values():
            page.parent = None

    @classmethod
    def init_connection(cls, destination):
        """反向 BFS:给每个能到达 ``destination`` 的页面记下 parent。

        与 ALAS 的 ``Page.init_connection`` 同构:从目标页出发,凡是"能一步跳到
        已访问页面"的页面都成为它的子节点,parent 指回更靠近目标的那一页。
        """
        cls.clear_connection()
        visited = {destination}
        while True:
            added = set()
            for page in visited:
                for candidate in cls.all_pages.values():
                    # links 以页面名称为键,不能拿 Page 对象去查。
                    if candidate in visited or page.name not in candidate.links:
                        continue
                    candidate.parent = page
                    added.add(candidate)
            if not added:
                return
            visited |= added

    def next_button(self):
        """朝目标页走一步要点的规则 id;已在目标页或不可达时为 None。"""
        if self.parent is None:
            return None
        return self.links.get(self.parent.name)


def build(rules, *, hub=MAIN, allowed_actions=None):
    """由规则列表推导页面图。

    - 页面:出现在**跨页点击规则**里的 ``stage`` / ``next_stage``,加上 hub。
      只做 ``observe`` 的状态(``*_owned``)不是页面;``*`` 表示任意页,不建点。
    - 检测按钮:子页用"只在这一页出现的点击规则";hub 用"观察后回到 main"的规则
      (它们只在主画面成立)。
    - 导航边:``stage --rule--> next_stage``,只收点击类,且不收 interrupt
      (任务弹窗是中断,不是导航)。
    """
    # "返回主画面"的观察阶段(achievements_return / mailbox_return / guild_return …)
    # 不是独立页面,而是**主画面到齐的证明**。带自环点击规则的阶段(如
    # achievements_fold)也算别名:自环只表示"还在这一页",不构成可导航的去处。
    def allowed(rule):
        return navigable(rule) and (allowed_actions is None or rule['id'] in allowed_actions)

    movers = {(rule.get('stage', hub), rule.get('next_stage')) for rule in rules
              if action(rule) == 'click' and allowed(rule)
              and rule.get('next_stage') not in (None, '*')
              and rule.get('next_stage') != rule.get('stage', hub)}
    from_stages = {stage for stage, _ in movers}
    aliases = set()
    for rule in rules:
        stage = rule.get('stage', hub)
        if (stage not in ('*', hub) and stage not in from_stages
                and rule.get('next_stage') == hub and action(rule) == 'observe'):
            aliases.add(stage)

    def resolve(name):
        return hub if name in aliases else name

    stages = {hub}
    checks = {hub: []}
    own_checks: dict[str, list[str]] = {}
    for rule in rules:
        stage = rule.get('stage', hub)
        if stage == '*':
            continue
        if rule.get('next_stage') == hub and action(rule) == 'observe':
            checks[hub].append(rule['id'])
        if action(rule) != 'click':
            continue
        stages.add(resolve(stage))
        own_checks.setdefault(resolve(stage), []).append(rule['id'])
        if rule.get('next_stage') not in (None, '*'):
            stages.add(resolve(rule['next_stage']))

    Page.clear()
    pages = {}
    for name in sorted(stages):
        pages[name] = Page(name, checks.get(name, ()) if name == hub
                           else own_checks.get(name, ()))

    for rule in rules:
        nxt = rule.get('next_stage')
        stage = rule.get('stage', hub)
        if not nxt or nxt == '*' or stage == '*':
            continue
        source, target = resolve(stage), resolve(nxt)
        if source not in pages or target not in pages or source == target:
            continue
        if action(rule) == 'observe':
            # 观察类规则不发输入就推进 stage(例如 guild_material_skip),
            # 这类边记为"等待即可",导航时点不了也不需要点。
            pages[source].links.setdefault(target, None)
        elif action(rule) == 'click' and not rule.get('interrupt') and allowed(rule):
            # 同一目标只保留配置里第一条边(例如 mailbox_close / limit_close
            # 都指向返回主画面),后出现的不覆盖,结果由配置顺序决定。
            pages[source].links.setdefault(target, rule['id'])
    return pages
