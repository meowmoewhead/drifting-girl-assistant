"""非阻塞页面导航:每帧只产出一个"朝目标走一步"的点击,交给主循环执行。

ALAS 的 ``ui_goto`` 自带阻塞截图循环;本项目的主循环是单帧驱动的(证据、点击计数、
Maa 回调都在同一帧内完成),所以这里把导航拆成"每帧一步":

    nav = Navigator(pages, appear=lambda rule_id: module.appear(...))
    step = nav.step(frame)          # 只做匹配,不发输入
    if step.kind == 'click':
        ... 交给既有候选执行路径点击 step.button
    elif step.kind == 'arrived':
        ... 收尾

这样"下一步是谁"由页面图算出来,而不是写死在 stage 里 —— 画面变化不再直接等于失败。
"""

from __future__ import annotations

from .pages import MAIN, Page


def nav_requested(rules, stage):
    """该 stage 是否显式允许页面图导航(规则上写 ``ui_nav: true``)。

    默认关闭:只有明确标注的步骤才会在"没等到预期画面"时按页面图退回,
    其余步骤仍保持原来的停机语义,避免改变已经验证过的流程。
    """
    return any(rule.get('ui_nav') and rule.get('stage') == stage for rule, _, _ in rules)


class Step:
    """一步导航的结果。

    ``kind``:
      - ``arrived`` 单帧目标页提示，仍须调用方连续确认;
      - ``click``   需要点击 ``button``;
      - ``wait``    这一步不用输入(观察类规则会自己推进 stage),等下一帧即可;
      - ``no_route`` 图上无路；调用方可继续已授权的有界关闭流程，不能猜点;
      - ``unknown`` 画面不属于任何已知页面。
    """

    __slots__ = ('kind', 'page', 'button', 'reason')

    def __init__(self, kind, page=None, button=None, reason=''):
        self.kind = kind
        self.page = page
        self.button = button
        self.reason = reason

    def __bool__(self):
        return self.kind == 'click'

    def __eq__(self, other):
        return (isinstance(other, Step) and self.kind == other.kind
                and self.page == other.page and self.button == other.button)

    def __repr__(self):
        return f'Step({self.kind}, page={self.page}, button={self.button})'


class Navigator:
    def __init__(self, pages, appear, *, hub=MAIN):
        self.pages = pages
        self.appear = appear
        self.hub = hub
        self.current = None

    def _page(self, name):
        return self.pages.get(name)

    def page_visible(self, page):
        """页面上任意一个检测规则命中即认为在该页。"""
        if page is None:
            return False
        return any(self.appear(rule_id) for rule_id in page.check)

    def detect(self, prefer=()):
        """识别当前页:优先 ``prefer`` 里的页,其次非 hub 页,最后 hub。

        非 hub 优先是刻意的:主画面的检测规则是"已返回主画面"的观察规则,
        若同时还能看到一个子页面,说明其实还在子页面里,不能误判成已回主界面。
        """
        ordered = [self._page(name) for name in prefer if self._page(name) is not None]
        rest = [page for name, page in sorted(self.pages.items())
                if name not in prefer and name != self.hub]
        tail = [self._page(self.hub)] if self._page(self.hub) is not None else []
        for page in ordered + rest + tail:
            if self.page_visible(page):
                self.current = page
                return page
        self.current = None
        return None

    def step(self, destination=None, *, prefer=()):
        """算出一帧内的导航动作,不发任何输入。"""
        destination = destination or self._page(self.hub)
        if destination is None:
            return Step('no_route', None, None, '目标页面未定义')
        page = self.detect(prefer)
        if page is None:
            return Step('unknown', None, None, '画面不属于任何已知页面')
        if page == destination:
            return Step('arrived', page)
        Page.init_connection(destination)
        button = page.next_button()
        if page.parent is not None and button is None:
            return Step('wait', page, None, f'{page} 等观察推进到 {page.parent}')
        if button is None:
            return Step('no_route', page, None, f'{page} 无法到达 {destination}')
        return Step('click', page, button, f'{page} → {page.parent}')

    def route(self, destination=None):
        """当前页到目标页要依次点击的规则 id 列表(只用于日志与测试)。"""
        destination = destination or self._page(self.hub)
        page = self.current if self.current is not None else self.detect()
        if page is None or destination is None:
            return []
        Page.init_connection(destination)
        buttons = []
        seen = set()
        while page is not None and page != destination and page.name not in seen:
            seen.add(page.name)
            button = page.next_button()
            if page.parent is None:
                return []
            if button is not None:
                buttons.append(button)
            page = page.parent
        return buttons


