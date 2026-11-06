"""页面图与导航步骤的单元测试(ALAS page link + 反向 BFS 的等价实现)。"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.core.pages import MAIN, Page, build
from drift.core.ui import Navigator, Step, nav_requested


def rule(rid, stage, next_stage, action='click', interrupt=False):
    return dict(id=rid, name=rid, stage=stage, next_stage=next_stage,
                action=action, **({'interrupt': True} if interrupt else {}))


SAMPLE = [
    rule('main_open_mail', MAIN, 'mailbox'),
    rule('mail_claim', 'mailbox', 'mailbox_done'),
    rule('mail_close', 'mailbox_done', 'mailbox_return'),
    rule('mail_returned', 'mailbox_return', MAIN, action='observe'),
    rule('main_open_panel', MAIN, 'panel'),
    rule('panel_close', 'panel', 'panel_return'),
    rule('panel_returned', 'panel_return', MAIN, action='observe'),
    rule('popup_close', '*', 'popup_return', interrupt=True),
    rule('popup_returned', 'popup_return', MAIN, action='observe'),
]


class PageGraphTests(unittest.TestCase):
    def setUp(self):
        Page.clear()
        self.pages = build(SAMPLE)

    def test_return_stages_become_hub_aliases(self):
        self.assertIn(MAIN, self.pages)
        self.assertNotIn('mailbox_return', self.pages)
        self.assertNotIn('panel_return', self.pages)
        self.assertNotIn('popup_return', self.pages)
        checks = self.pages[MAIN].check
        self.assertIn('mail_returned', checks)
        self.assertIn('panel_returned', checks)

    def test_links_are_taken_from_click_rules_only(self):
        self.assertEqual(self.pages[MAIN].links['mailbox'], 'main_open_mail')
        self.assertEqual(self.pages['mailbox'].links['mailbox_done'], 'mail_claim')
        self.assertEqual(self.pages['mailbox_done'].links[MAIN], 'mail_close')

    def test_interrupt_rules_are_not_navigation_edges(self):
        for page in self.pages.values():
            self.assertNotIn('popup_close', page.links.values())

    def test_links_land_on_hub_not_on_alias(self):
        self.assertEqual(self.pages['mailbox_done'].links, {MAIN: 'mail_close'})

    def test_init_connection_walks_back_toward_destination(self):
        Page.init_connection(self.pages[MAIN])
        self.assertEqual(self.pages['mailbox_done'].parent, self.pages[MAIN])
        self.assertEqual(self.pages['mailbox'].parent, self.pages['mailbox_done'])
        self.assertIsNone(self.pages[MAIN].parent)
        self.assertEqual(self.pages['mailbox'].next_button(), 'mail_claim')
        self.assertEqual(self.pages['mailbox_done'].next_button(), 'mail_close')
        self.assertIsNone(self.pages[MAIN].next_button())

    def test_unreachable_page_has_no_route(self):
        pages = build([rule('orphan_click', 'orphan', 'nowhere')])
        Page.init_connection(pages[MAIN])
        self.assertIsNone(pages['orphan'].parent)
        self.assertIsNone(pages['orphan'].next_button())

    def test_cycles_terminate(self):
        pages = build([
            rule('a_to_b', 'a', 'b'), rule('b_to_a', 'b', 'a'),
            rule('main_to_a', MAIN, 'a'), rule('a_home', 'a', 'a_return'),
            rule('a_returned', 'a_return', MAIN, action='observe'),
        ])
        Page.init_connection(pages[MAIN])
        self.assertEqual(len(Page.iter_pages()), len(pages))
        self.assertEqual(pages['a'].parent, pages[MAIN])

    def test_star_stage_is_not_a_page(self):
        self.assertNotIn('*', self.pages)


class NavigatorTests(unittest.TestCase):
    def setUp(self):
        Page.clear()
        self.pages = build(SAMPLE)

    def navigator(self, visible):
        seen = []
        def appear(rid):
            seen.append(rid)
            return rid in visible
        nav = Navigator(self.pages, appear)
        nav.seen = seen
        return nav

    def test_arrived_when_destination_visible(self):
        nav = self.navigator({'mail_returned'})
        step = nav.step()
        self.assertEqual(step, Step('arrived', self.pages[MAIN]))

    def test_unknown_screen_never_produces_a_click(self):
        nav = self.navigator(set())
        step = nav.step()
        self.assertEqual(step.kind, 'unknown')
        self.assertIsNone(step.button)
        self.assertFalse(step)

    def test_click_targets_the_edge_toward_hub(self):
        nav = self.navigator({'mail_claim'})
        step = nav.step()
        self.assertEqual(step.kind, 'click')
        self.assertEqual(step.page, self.pages['mailbox'])
        self.assertEqual(step.button, 'mail_claim')

    def test_child_page_wins_over_hub_check(self):
        # 主画面的返回证明与子页面按钮同时可见时,不能误判成已在主界面。
        nav = self.navigator({'mail_returned', 'mail_close'})
        step = nav.step()
        self.assertEqual(step.page, self.pages['mailbox_done'])
        self.assertEqual(step.button, 'mail_close')

    def test_prefer_wins(self):
        nav = self.navigator({'mail_claim', 'main_open_mail'})
        step = nav.step(prefer=('mailbox',))
        self.assertEqual(step.page, self.pages['mailbox'])

    def test_no_route_reports_instead_of_guessing(self):
        pages = build([rule('orphan_click', 'orphan', 'nowhere')])
        nav = Navigator(pages, lambda rid: rid == 'orphan_click')
        step = nav.step()
        self.assertEqual(step.kind, 'no_route')
        self.assertIsNone(step.button)

    def test_route_lists_buttons_to_destination(self):
        nav = self.navigator({'mail_claim'})
        self.assertEqual(nav.route(), ['mail_claim', 'mail_close'])

    def test_route_is_empty_when_unknown(self):
        nav = self.navigator(set())
        self.assertEqual(nav.route(), [])


class NavRequestTests(unittest.TestCase):
    def test_off_unless_a_rule_for_this_stage_opts_in(self):
        rules = [(rule('a', 'mailbox', 'main'), None, 0)]
        self.assertFalse(nav_requested(rules, 'mailbox'))
        self.assertFalse(nav_requested(rules, 'main'))

    def test_flag_on_the_matching_stage_only(self):
        flagged = rule('a', 'mailbox', 'main')
        flagged['ui_nav'] = True
        rules = [(flagged, None, 0), (rule('b', 'main', 'mailbox'), None, 0)]
        self.assertTrue(nav_requested(rules, 'mailbox'))
        self.assertFalse(nav_requested(rules, 'main'))

    def test_truthy_only_counts_as_enabled(self):
        flagged = rule('a', 'mailbox', 'main')
        flagged['ui_nav'] = 0
        self.assertFalse(nav_requested([(flagged, None, 0)], 'mailbox'))

    def test_real_config_has_no_navigation_enabled_by_default(self):
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        rules = [(item, None, 0) for item in config['rules']]
        stages = {item.get('stage') for item in config['rules'] if item.get('stage')}
        self.assertFalse([stage for stage in stages if nav_requested(rules, stage)])


class RealConfigTests(unittest.TestCase):
    """用真实 config.json 验证推导出的图能覆盖我们踩过坑的那几条返回路径。"""

    @classmethod
    def setUpClass(cls):
        Page.clear()
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.pages = build(config['rules'])

    def test_expected_pages_exist(self):
        for name in (MAIN, 'mailbox', 'mailbox_done', 'achievements_tab',
                     'achievements_claim', 'daily_tab', 'daily_reward', 'guild_tab'):
            self.assertIn(name, self.pages)

    def test_mailbox_and_crown_returns_have_routes(self):
        Page.init_connection(self.pages[MAIN])
        # 同一目标可能有多条等价边(普通关闭 / 领取上限关闭),任一条都算有路。
        expected = {'mailbox_done': {'driftinggirl_mailbox_close',
                                     'driftinggirl_mailbox_limit_close'},
                    'achievements_claim': {'driftinggirl_achievements_close'}}
        for name, buttons in expected.items():
            page = self.pages[name]
            self.assertIsNotNone(page.parent, f'{name} 无法回到主界面')
            self.assertIn(page.next_button(), buttons)

    def test_only_purchase_screens_lack_a_route(self):
        # 购买数量/确认页只有"买"和"取消"两条路,我们不建模取消,
        # 所以它们必须报 no_route,而不是给出一条会花钱的导航边。
        purchase_screens = {'guild_material_confirm', 'guild_material_quantity',
                            'guild_potion_confirm', 'guild_potion_quantity'}
        Page.init_connection(self.pages[MAIN])
        unreachable = {name for name, page in self.pages.items()
                       if name != MAIN and page.parent is None}
        self.assertEqual(unreachable, purchase_screens)
        for name in purchase_screens:
            self.assertIsNone(self.pages[name].next_button())

    def test_observe_edges_navigate_without_input(self):
        # guild_material_skip 是 observe:它自己推进 stage,导航这一步只需等待。
        Page.init_connection(self.pages[MAIN])
        nav = Navigator(self.pages, lambda rid: False)
        step = nav.step()
        self.assertEqual(step.kind, 'unknown')
        nav.current = self.pages['guild_material']
        self.assertEqual(nav.route(), ['driftinggirl_guild_close'])

    def test_routes_never_contain_purchase_actions(self):
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        rules = {rule['id']: rule for rule in config['rules']}
        Page.init_connection(self.pages[MAIN])
        for name, page in self.pages.items():
            nav = Navigator(self.pages, lambda rid: False)
            nav.current = page
            for button in nav.route():
                rule = rules[button]
                self.assertFalse(any(rule.get(key) for key in
                                     ('purchase_condition', 'quantity_condition',
                                      'confirmation_condition')),
                                 f'{name} 的导航路包含了购买动作 {button}')
                self.assertNotIn('_buy', button)


if __name__ == '__main__':
    unittest.main()
