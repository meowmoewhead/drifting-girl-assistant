"""Restaurant preferences: offline state-machine, API, factory and panel checks."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from drift.perception.restaurant import CATEGORIES
from drift.tasks.features import FEATURE_BY_ID, feature_states
from drift.tasks.restaurant_upgrade import RESTAURANT_ID, RestaurantUpgrade, pack
from tests import test_clicker as helpers

ROOT = Path(__file__).resolve().parents[1]


class RestaurantTargetTests(unittest.TestCase):
    def setUp(self):
        # These tests isolate category routing; real fresh-frame motion gates
        # are exercised separately by test_restaurant_list and real replays.
        self.addCleanup(patch.stopall)
        patch('drift.tasks.restaurant_list.settled',return_value=True).start()

    @staticmethod
    def cover_category(task,category):
        task.journal['visited'].extend(dict(category=category,level=20,
            art=pack(np.random.default_rng(category*10+i).integers(0,256,(8,8,3),dtype=np.uint8))) for i in range(9))

    def controller(self, folder, **options):
        task = RestaurantUpgrade(ROOT, None, Mock(return_value=True), Mock(), 'test',
                                 state_path=Path(folder)/'state.json', clock=lambda:100,
                                 **options)
        task.root = Path(folder)
        task.same_world = Mock(return_value=True)
        task.visual = Mock()
        task.visual.point.return_value = (600, 300)
        task.visual.list_key.return_value = np.zeros((5, 5), dtype=np.uint8)
        task.visual.cards.return_value = []
        task.visual.art_matches.side_effect=np.array_equal
        if not task.active:
            task.phase = 'atlas'
            task.journal.update(pending=True, phase='atlas', deadline=1000,
                                phase_deadline=200, attempts={}, categories=[],
                                visited=[], skipped=[], upgraded=0, submitted=0)
        return task

    def confirm(self, task, category):
        task.visual.category.return_value = category
        # Preserve the existing confirmation timing, including index zero.
        for now in (100, 100.4, 100.8, 103):
            task.clock = lambda now=now:now
            task.step(object(), due=False)

    def test_missing_none_empty_and_all_keep_identical_default_journal_and_events(self):
        for category in range(9):
            baseline = None
            for options in ({}, {'targets':None}, {'targets':[]}, {'targets':list(CATEGORIES)}):
                with self.subTest(category=category, options=options), tempfile.TemporaryDirectory() as d:
                    task = self.controller(d, **options)
                    self.confirm(task, category)
                    self.assertEqual(task.phase, 'align')
                    task.action.assert_not_called()
                    actual = (task.path.read_bytes(), task.event.call_args_list)
                    if baseline is None:
                        baseline = actual
                    else:
                        self.assertEqual(actual, baseline)
                    self.assertNotIn('skipped_categories', task.journal)

    def test_each_unselected_category_skips_without_inspecting_cards_or_spending(self):
        for category, name in enumerate(CATEGORIES):
            with self.subTest(category=name), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=[CATEGORIES[(category+1) % 9]])
                self.confirm(task, category)
                self.assertEqual(task.phase, 'category_next')
                task.action.assert_not_called()
                task.visual.list_key.assert_not_called()
                task.visual.cards.assert_not_called()
                task.visual.detail_values.assert_not_called()
                task.event.assert_any_call(f'餐品强化：跳过「{name}」（不在已选类别内）')
                self.assertEqual(task.journal['categories'], [])
                self.assertEqual(task.status()['checked'], 0)
                self.assertEqual(task.status()['upgraded'], 0)
                self.assertEqual(task.status()['submitted'], 0)
                task.clock = lambda:104
                task.step(object(), due=False)
                self.assertEqual(task.action.call_args.args[0]['kind'], f'next:{category}')

    def test_selected_categories_keep_card_feed_and_upgrade_paths(self):
        for category in range(9):
            for ready in (False, True):
                with self.subTest(category=category, ready=ready), tempfile.TemporaryDirectory() as d:
                    task = self.controller(d, targets=[CATEGORIES[category]])
                    self.confirm(task, category)
                    self.assertEqual(task.phase, 'align')
                    task.journal['aligns'] = 4
                    task._confirmed = Mock(return_value=True)
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'list')
                    task.visual.cards.return_value = [dict(point=(200, 400), level=1,
                                                          art=np.zeros((4, 4, 3), dtype=np.uint8))]
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'detail')
                    task.clock = lambda:105
                    task.visual.detail_values.return_value = dict(identity='dish', level=1,
                        counters=[(8 if ready else 0, 8)]*4)
                    task.visual.upgrade_ready.return_value = ready
                    task.visual.ingredient_ready.return_value = True
                    task.step(object(), due=False)
                    kinds = [call.args[0]['kind'] for call in task.action.call_args_list]
                    self.assertTrue(kinds[0].startswith(f'dish:{category}:'))
                    self.assertTrue(kinds[1].startswith('upgrade:' if ready else 'feed:'))
                    self.assertEqual(task.phase, 'upgrade_wait' if ready else 'feed_wait')

    def test_unknown_or_unconfirmed_category_does_not_skip_or_click(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d, targets=['烧烤'])
            self.confirm(task, None)
            self.assertEqual(task.phase, 'atlas')
            task.visual.category.return_value = 0
            task.clock = lambda:104
            task.step(object(), due=False)
            self.assertEqual(task.phase, 'atlas')
            task.action.assert_not_called()
            task.event.assert_not_called()

    def test_skip_log_once_per_run_even_after_revisit_and_restart(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d, targets=['烧烤'])
            self.confirm(task, 3)
            task._set('atlas')
            self.confirm(task, 3)
            message = '餐品强化：跳过「油炸」（不在已选类别内）'
            self.assertEqual([c.args[0] for c in task.event.call_args_list].count(message), 1)
            resumed = self.controller(d, targets=['烧烤'])
            resumed._set('atlas')
            self.confirm(resumed, 3)
            self.assertNotIn(message, [c.args[0] for c in resumed.event.call_args_list])
            resumed.journal['pending'] = False
            resumed.phase = 'idle'
            resumed.start_screen = Mock(return_value={'route':False,'parallel':False,'header':['rank','world'],'at_restaurant':True})
            resumed.origin_screen = Mock(return_value={'route':False,'at_restaurant':True,'parallel':False,'header':['rank','world']})
            resumed._confirmed = Mock(return_value=True)
            resumed.view._battle=Mock(return_value=False)
            resumed.step(object())
            resumed._set('atlas')
            resumed.step(object(), due=False)
            self.assertEqual([c.args[0] for c in resumed.event.call_args_list].count(message), 1)

    def test_mixed_run_stops_after_selected_from_any_start_without_counting_skips_as_completed(self):
        for start in range(9):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=['干货', '烧烤'])
                task._confirmed = Mock(return_value=True)
                completed = []
                for index in range(9):
                    category = (start+index) % 9
                    task.clock = lambda index=index:100+index*5
                    task.visual.category.return_value = category
                    if index:
                        task.step(object(), due=False)
                        self.assertEqual(task.phase, 'atlas')
                    task.step(object(), due=False)
                    if category in (0, 8):
                        self.assertEqual(task.phase, 'align')
                        task.journal['aligns'] = 4
                        task.step(object(), due=False)
                        self.cover_category(task,category)
                        task.step(object(), due=False)
                        completed.append(category)
                    self.assertEqual(task.phase, 'category_next')
                    self.assertEqual(task.journal['categories'], completed)
                    task.step(object(), due=False)
                    finished = set(completed) == {0, 8}
                    self.assertEqual(task.phase, 'close' if finished else 'category_next')
                    if finished:
                        break
                actions = [c.args[0]['kind'] for c in task.action.call_args_list]
                self.assertEqual(len(actions), index)
                self.assertTrue(all(k.startswith('next:') for k in actions))
                self.assertEqual(task.status()['checked'],9*len(completed))
                skips = [c.args[0] for c in task.event.call_args_list if '不在已选类别内' in c.args[0]]
                self.assertEqual(len(skips), index-1)

    def test_single_selection_only_opens_selected_dishes_from_every_start(self):
        for selected in range(9):
            for start in range(9):
                with self.subTest(selected=selected, start=start), tempfile.TemporaryDirectory() as d:
                    task = self.controller(d, targets=[CATEGORIES[selected]])
                    task._confirmed = Mock(return_value=True)
                    for index in range(9):
                        category = (start+index) % 9
                        task.clock = lambda index=index:100+index*3
                        task.visual.category.return_value = category
                        if index:
                            task.step(object(), due=False)
                        task.step(object(), due=False)
                        if category == selected:
                            self.assertEqual(task.phase, 'align')
                            break
                        self.assertEqual(task.phase, 'category_next')
                        task.visual.cards.assert_not_called()
                        task.visual.detail_values.assert_not_called()
                        task.step(object(), due=False)
                    task.journal['aligns'] = 4
                    task.step(object(), due=False)
                    task.visual.cards.return_value = [dict(point=(200, 400), level=1,
                                                          art=np.zeros((4, 4, 3), dtype=np.uint8))]
                    task.step(object(), due=False)
                    kinds = [call.args[0]['kind'] for call in task.action.call_args_list]
                    self.assertEqual(len(kinds), index+1)
                    self.assertTrue(all(kind.startswith('next:') for kind in kinds[:-1]))
                    self.assertTrue(kinds[-1].startswith(f'dish:{selected}:'))
                    self.assertEqual(task.journal.get('skipped_categories', []),
                                     [(start+n) % 9 for n in range(index)])

    def test_selected_completed_category_closes_without_visiting_other_eight(self):
        for category in range(9):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=[CATEGORIES[category]])
                task.journal.update(category=category, categories=[category])
                task.visual.category.return_value = category
                task._set('category_next')
                task.step(object(), due=False)
                self.assertEqual(task.phase, 'close')
                task.action.assert_not_called()
                self.assertNotIn('skipped_categories', task.journal)

    def test_completed_category_revisit_does_not_reopen_or_inspect_dishes(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d, targets=['干货', '烧烤'])
            task.journal['categories'] = [0]
            self.confirm(task, 0)
            self.assertEqual(task.phase, 'category_next')
            task.visual.cards.assert_not_called()
            task.visual.list_key.assert_not_called()
            task.action.assert_not_called()
            task.clock = lambda:104
            task.step(object(), due=False)
            self.assertEqual(task.action.call_args.args[0]['kind'], 'next:0')

    def test_all_option_walks_all_nine_before_closing_from_every_start(self):
        for start in range(9):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=[])
                task._confirmed = Mock(return_value=True)
                for index in range(9):
                    category = (start+index) % 9
                    task.clock = lambda index=index:100+index*3
                    task.visual.category.return_value = category
                    if index:
                        task.step(object(), due=False)
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'align')
                    task.journal['aligns'] = 4
                    task.step(object(), due=False)
                    self.cover_category(task,category)
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'category_next')
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'close' if index == 8 else 'category_next')
                self.assertEqual(set(task.journal['categories']), set(range(9)))
                self.assertNotIn('skipped_categories', task.journal)
                self.assertEqual([call.args[0]['kind'] for call in task.action.call_args_list],
                                 [f'next:{(start+n) % 9}' for n in range(8)])

    def unavailable_dish(self, task, identity, *, full=False):
        task.visual.detail_values.return_value = dict(identity=identity, level=1,
            counters=[(8 if full else 0, 8)]*4)
        task.visual.upgrade_ready.return_value = False
        task.visual.ingredient_ready.return_value = False
        task.journal.update(card=dict(level=1, art=pack(np.zeros((4, 4, 3), dtype=np.uint8))),
                            feed_counts={}, blocked_ingredients=[])
        task._set('detail', 120)
        task.step(object(), due=False)
        self.assertEqual(task.phase, 'detail_close')
        self.assertIn('食材不足、已交满', task.journal['skipped'][-1]['reason'])
        task.visual.detail.return_value = False
        task.visual.atlas.return_value = True
        task.step(object(), due=False)

    def test_all_nine_unavailable_dishes_close_without_scroll_then_restore_and_finish(self):
        for running in (False, True):
            with self.subTest(running=running), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=['烧烤'])
                task._confirmed = Mock(return_value=True)
                task.journal.update(category=8, origin={'at_restaurant':True, 'route':running})
                for index in range(9):
                    self.unavailable_dish(task, f'dish-{index}', full=index % 2 == 0)
                    self.assertEqual(task.phase, 'category_next' if index == 8 else 'list')
                task.step(object(), due=False)
                self.assertEqual(task.phase, 'close')
                self.assertEqual(task.journal['categories'], [8])
                self.assertEqual(len(task.journal['skipped']), 9)
                task.action.assert_not_called()
                task.visual.list_key.assert_not_called()

                # Exercise the real close / restore / _finish path, never a mocked finish.
                task.visual.panel.return_value = True
                task.step(object(), due=False)
                self.assertEqual(task.action.call_args.args[0]['kind'], 'panel-close')
                task.clock = lambda:102
                task.visual.panel.return_value = False
                task.at_restaurant = Mock(return_value=True)
                task.step(object(), due=False)
                self.assertEqual(task.phase, 'restore')
                self.assertTrue(task.active)
                task.fishing = Mock()
                task.fishing._main_visible.return_value = True
                task.sailing = Mock()
                task.sailing.running.return_value = False
                if running:
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'restore_menu')
                    task.journal['settings'] = {'original':'unchanged'}
                    task.settings = Mock(return_value=task.journal['settings'])
                    task.clock = lambda:104
                    task.step(object(), due=False)
                    self.assertEqual(task.phase, 'restore_verify')
                    task.clock = lambda:106
                    task.sailing.running.return_value = True
                    task.fishing.blue_fill.return_value = False
                    task.fishing.real_battle.return_value = False
                    task._confirmed = lambda key, *args: key is not None
                    task.step(object(), due=False)
                    self.assertTrue(task.active)
                    task.fishing.blue_fill.return_value = True
                task.step(object(), due=False)
                self.assertIs(task.done, True)
                self.assertFalse(task.active)
                saved = json.loads(task.path.read_text(encoding='utf-8'))
                self.assertIs(saved['success'], True)
                self.assertEqual(saved['phase'], 'idle')
                self.assertEqual((saved['submitted'], saved['upgraded']), (0, 0))
                self.assertEqual([call.args[0]['kind'] for call in task.action.call_args_list],
                                 ['panel-close', 'restore-open', 'restore-start'] if running else ['panel-close'])

    def test_one_unavailable_dish_does_not_hide_another_feedable_or_upgradeable_dish(self):
        for upgrade in (False, True):
            with self.subTest(upgrade=upgrade), tempfile.TemporaryDirectory() as d:
                task = self.controller(d, targets=['烧烤'])
                task._confirmed = Mock(return_value=True)
                task.visual.art_matches.return_value = False
                task.visual.category.return_value = 8
                task.journal.update(category=8, scrolls=0, previous_list=None)
                self.unavailable_dish(task, 'no-material')
                self.assertEqual(task.phase, 'list')
                task.visual.cards.return_value = [dict(point=(200, 400), level=1,
                                                      art=np.ones((4, 4, 3), dtype=np.uint8))]
                task.step(object(), due=False)
                self.assertEqual(task.phase, 'detail')
                task.clock = lambda:102
                task.visual.detail_values.return_value = dict(identity='another', level=1,
                    counters=[(8 if upgrade else 0, 8)]*4)
                task.visual.upgrade_ready.return_value = upgrade
                task.visual.ingredient_ready.return_value = True
                task.step(object(), due=False)
                self.assertEqual(task.phase, 'upgrade_wait' if upgrade else 'feed_wait')
                self.assertTrue(task.action.call_args.args[0]['kind'].startswith('upgrade:' if upgrade else 'feed:'))

    def test_duplicate_or_other_category_skips_do_not_prove_exhaustion(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d, targets=['烧烤'])
            task._confirmed = Mock(return_value=True)
            task.journal['category'] = 0
            for index in range(8):
                self.unavailable_dish(task, f'other-{index}')
            task.journal['category'] = 8
            for index in range(9):
                self.unavailable_dish(task, 'same-dish')
                self.assertEqual(task.phase, 'list')
            self.assertEqual(task.journal['categories'], [])
            task.action.assert_not_called()

    def test_short_list_end_requires_backscan_and_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            task=self.controller(d,targets=['干货'])
            task._confirmed=Mock(return_value=True)
            task.visual.category.return_value=0
            task.journal.update(category=0,scrolls=5,aligns=4)
            task._set('list',1200)
            task.step(object(),due=False)
            self.assertEqual(task.phase,'list')
            self.assertEqual(task.journal['atlas_scan']['direction'],'backward')
            self.assertEqual(task.journal['categories'],[])
            task.journal['atlas_scan']['back_steps']=5
            task.step(object(),due=False)
            self.assertEqual(task.phase,'category_next')
            self.assertEqual(task.journal['categories'],[])
            self.assertEqual(len(task.journal['incomplete_categories']),1)
            task.step(object(),due=False)
            self.assertEqual(task.phase,'close')
            task.action.assert_not_called()

    def test_revisited_category_cannot_complete_run_early(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d, targets=['烧烤'])
            self.confirm(task, 0)
            for category in [0]*9 + list(range(1, 8)):
                task._set('atlas')
                self.confirm(task, category)
            task.clock = lambda:105
            task.step(object(), due=False)
            self.assertEqual(task.phase, 'category_next')

    def test_legacy_completed_journal_without_preference_fields_still_closes(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.controller(d)
            task.journal.update(categories=list(range(9)), category=8)
            task._set('category_next')
            task.step(object(), due=False)
            self.assertEqual(task.phase, 'close')
            task.action.assert_not_called()


class RestaurantPreferenceAPITests(unittest.TestCase):
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule

    def setUp(self):
        helpers.FrameworkTests.setUp(self)
        self.rule().update(id=RESTAURANT_ID, stage='restaurant_upgrade_owned', interval=259200.0)
        self.app.persist()

    def test_targets_saved_deduplicated_and_returned_without_changing_other_rules(self):
        self.rule()
        unrelated = deepcopy(self.app.config['rules'][-1])
        selected = ['烧烤', '干货', '烧烤']
        result = self.app.change_feature({'id':'restaurant_upgrade', 'targets':selected})
        saved = json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.assertEqual(set(result['targets']), {'干货', '烧烤'})
        self.assertEqual(len(result['targets']), 2)
        self.assertEqual(saved['rules'][0]['targets'], result['targets'])
        self.assertEqual(saved['rules'][-1], unrelated)
        self.assertEqual(selected, ['烧烤', '干货', '烧烤'])
        fresh = next(f for f in feature_states(saved) if f['id'] == 'restaurant_upgrade')
        self.assertEqual(fresh['targets'], result['targets'])

    def test_empty_list_resets_selection_to_all(self):
        self.app.change_feature({'id':'restaurant_upgrade', 'targets':['烧烤']})
        result = self.app.change_feature({'id':'restaurant_upgrade', 'targets':[]})
        self.assertEqual(result['targets'], [])
        self.assertEqual(self.app.config['rules'][0]['targets'], [])

    def test_missing_targets_preserves_existing_config_bytes_and_response_shape(self):
        before = (self.root/'config.json').read_bytes()
        result = self.app.change_feature({'id':'restaurant_upgrade', 'interval':259200})
        self.assertEqual((self.root/'config.json').read_bytes(), before)
        self.assertNotIn('targets', result)
        self.app.config['rules'][0]['targets'] = ['汤']
        self.app.persist()
        result = self.app.change_feature({'id':'restaurant_upgrade', 'interval':123})
        self.assertEqual(result['targets'], ['汤'])

    def test_invalid_targets_are_atomic_and_rejected_as_category_errors(self):
        invalid = [None, '', '干货', 1, True, {}, (), ('干货',), {'干货'},
                   [1], [True], [None], [[]], [{}], ['干货', '武器攻击'], ['干货 ']]
        before = deepcopy(self.app.config)
        raw = (self.root/'config.json').read_bytes()
        revision = self.app.config_revision
        for targets in invalid:
            with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, '类别'):
                self.app.change_feature({'id':'restaurant_upgrade', 'enabled':False, 'targets':targets})
            self.assertEqual(self.app.config, before)
            self.assertEqual((self.root/'config.json').read_bytes(), raw)
            self.assertEqual(self.app.config_revision, revision)

    def test_targets_rejected_for_every_other_feature(self):
        for feature in FEATURE_BY_ID.values():
            if feature['id'] == 'restaurant_upgrade':
                continue
            self.app.config['rules'].extend(dict(self.app.config['rules'][0], id=rid)
                                            for rid in feature['rules'])
        before = deepcopy(self.app.config)
        for feature in FEATURE_BY_ID:
            if feature != 'restaurant_upgrade':
                with self.subTest(feature=feature), self.assertRaisesRegex(ValueError, '仅用于餐品强化'):
                    self.app.change_feature({'id':feature, 'targets':[]})
        self.assertEqual(self.app.config, before)

    def test_targets_can_queue_while_running_without_calling_idle_guard(self):
        with patch.object(self.app, 'running', return_value=True):
            with patch.object(self.app, 'idle_only', wraps=self.app.idle_only) as idle:
                result=self.app.change_feature({'id':'restaurant_upgrade', 'targets':['烧烤']})
                idle.assert_not_called()
            self.assertTrue(self.app.status()['settings_pending'])
        self.assertEqual(result['targets'],['烧烤'])

    def test_persist_failure_keeps_original_preferences(self):
        before = deepcopy(self.app.config)
        with patch.object(self.app, 'persist', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.app.change_feature({'id':'restaurant_upgrade', 'targets':['烧烤']})
        self.assertEqual(self.app.config, before)


class RestaurantPreferenceFactoryTests(unittest.TestCase):
    def test_runtime_passes_rule_targets_to_controller_without_starting_runtime(self):
        tree = ast.parse((ROOT/'drift/runtime.py').read_text(encoding='utf-8-sig'))
        factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'make_restaurant')
        module = ast.Module(body=[factory], type_ignores=[])
        for options in ({}, {'targets':[]}, {'targets':['烧烤', '干货']}):
            with self.subTest(options=options):
                entry = dict(id=RESTAURANT_ID, enabled=True, interval=259200, **options)
                app = SimpleNamespace(root=ROOT, digit_reader=object(), event=Mock(),
                                      config={'serial':'test', 'rules':[entry]})
                constructor = Mock()
                scope = dict(self=app, rules=[(entry, None, None)], preview=False,
                             restaurant_click=Mock(), RestaurantUpgrade=constructor, RESTAURANT_ID=RESTAURANT_ID)
                exec(compile(module, '<isolated make_restaurant>', 'exec'), scope)
                self.assertIs(scope['make_restaurant'](), constructor.return_value)
                self.assertEqual(constructor.call_args.kwargs.get('targets'), options.get('targets'))


@unittest.skipUnless(shutil.which('node'), 'Node.js unavailable for offline panel execution')
class RestaurantPreferencePanelTests(unittest.TestCase):
    def panel(self, feature, selected=None):
        # Execute the actual renderDetail function with a minimal offline DOM.
        # No server, browser, API request or device is started.
        script = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map(), boxes = [], selects = [];
let markup = '', sent;
function element(id) {
  const node = {value:'', textContent:'', checked:false, listeners:{}, nextElementSibling:{},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    closest(selector) { return ['#settings-form','#schedule-form'].includes(selector) ? null : this; }, insertAdjacentHTML(position, text) { parse(text); }};
  Object.defineProperty(node, 'innerHTML', {set(text) { if (id === 'feature-detail') markup = text; parse(text); }});
  elements.set(id, node); return node;
}
function parse(text) {
  for (const match of text.matchAll(/\bid="([^"]+)"/g)) if (!elements.has(match[1])) element(match[1]);
  for (const match of text.matchAll(/<input\b[^>]*type="checkbox"[^>]*>/g)) {
    const tag = match[0], id = tag.match(/\bid="([^"]*)"/)?.[1], node = id ? elements.get(id) : element('box-' + boxes.length);
    node.value = tag.match(/\bvalue="([^"]*)"/)?.[1] || id || '';
    node.checked = /\schecked(?:\s|>|=)/.test(tag); boxes.push(node);
  }
  for (const match of text.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)) {
    const node = elements.get(match[1].match(/\bid="([^"]+)"/)[1]);
    node.multiple = /\bmultiple\b/.test(match[1]);
    node.options = Array.from(match[2].matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g), option => ({
      value:option[1].match(/\bvalue="([^"]*)"/)[1], label:option[2],
      selected:/\bselected\b/.test(option[1])}));
    node.value = (node.options.find(option=>option.selected) || node.options[0]).value;
    selects.push(node);
  }
}
element('feature-detail');
const context = {detailId:'', detailDirty:false, scheduleDirty:false, state:{rules:[],features:[input.feature]}, busy:false, connected:true,
  deviceScanning:false, deviceScan:null,
  $:id=>elements.get(id), esc:value=>String(value ?? ''), icon:()=>'', toggleMarkup:()=>'',
  notes:{}, executionKind:()=>'', nextRunText:()=>'', timeInputValue:()=>'',
  api:(path, data)=>{ sent = {path, data}; return Promise.resolve(); }, act:fn=>fn(),
  document:{querySelector:()=>({}), querySelectorAll:selector=> {
    if (selector.includes('#restaurant-targets')) return selector.includes(':checked') ? boxes.filter(b=>b.checked) : boxes;
    if (selector.includes('#feature-form')) return [...boxes, ...(selector.includes('#feature-form select') ? selects : [])];
    return [];
  }}};
const source = fs.readFileSync('web/app.js', 'utf8');
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function durationText('), source.indexOf('\nfunction notice(')), context);
vm.runInContext(source.slice(source.indexOf('function featureNote('), source.indexOf('\nfunction cruiseFields(')), context);
vm.runInContext(source.slice(source.indexOf('function cruiseFields('), source.indexOf('\nfunction route()')), context);
context.renderDetail(input.feature);
const select = selects[0];
const initial = select ? (select.value ? [select.value] : []) : boxes.filter(b=>b.checked).map(b=>b.value);
if (input.selected !== null) {
  if (select) { select.value = input.selected; select.listeners.change?.(); }
  else for (const box of boxes) { box.checked = input.selected.includes(box.value); box.listeners.input?.(); }
}
const dirty = context.detailDirty;
context.renderDetail(input.feature); // Status refresh must preserve unsaved choices.
elements.get('feature-form').onsubmit({preventDefault(){}});
for (const id of ['run', 'preview', 'stop', 'refresh', 'refresh-mini', 'device-detect','device-adb-path','device-select','device-connect','device-auto']) element(id);
vm.runInContext(source.slice(source.indexOf('function updateControls('), source.indexOf('\nfunction showNumbers(')), context);
const disabledStates = [];
for (const [running, busy, connected] of [[false,false,true], [true,false,true], [false,true,true], [false,false,false], [false,false,true]]) {
  context.state.running = running; context.busy = busy; context.connected = connected;
  context.updateControls(); disabledStates.push(select?.disabled ?? null);
}
process.stdout.write(JSON.stringify({markup, initial, dirty, sent,
  values:select ? select.options.map(option=>option.value) : boxes.map(b=>b.value),
  labels:select ? select.options.map(option=>option.label) : [],
  selectCount:selects.length, checkboxCount:boxes.length, multiple:select?.multiple, disabledStates}));
'''
        data = dict(feature={'id':'restaurant_upgrade', 'category':'', 'name':'', 'description':'',
                    'steps':[], 'rules':[], 'interval':259200, 'execution':{'mode':'timed'}, **feature}, selected=selected)
        result = subprocess.run([shutil.which('node'), '-e', script], cwd=ROOT,
                                input=json.dumps(data), text=True, encoding='utf-8',
                                capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_nine_categories_restore_selected_and_submit_changed_selection(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                result = self.panel({'targets':[category]}, '汤')
                self.assertEqual(result['values'], ['', *CATEGORIES])
                self.assertEqual(result['initial'], [category])
                self.assertTrue(result['dirty'])
                self.assertEqual(result['sent']['path'], '/api/features/change')
                self.assertEqual(result['sent']['data']['targets'], ['汤'])

    def test_all_option_means_all_and_is_explained(self):
        for feature in ({}, {'targets':[]}, {'targets':list(CATEGORIES)}):
            with self.subTest(feature=feature):
                result = self.panel(feature, '')
                self.assertEqual(result['initial'], [])
                self.assertEqual(result['sent']['data']['targets'], [])
                self.assertIn('全部', result['markup'])
                self.assertIn('九类', result['markup'])

    def test_dropdown_has_one_single_select_and_exactly_ten_options(self):
        result = self.panel({})
        self.assertEqual(result['selectCount'], 1)
        self.assertEqual(result['checkboxCount'], 0)
        self.assertIs(result['multiple'], False)
        self.assertEqual(result['values'], ['', *CATEGORIES])
        self.assertEqual(result['labels'], ['全部', *[f'{name}（{label}）' for name, label in zip(
            CATEGORIES, ('鱼饼获得量', '鱼竿攻击力', '鱼叉塔攻击力', '大炮攻击力',
                         '机关炮塔攻击力', '魔法石攻击力', '鲸鱼攻击力',
                         '琥珀/海蓝宝/次元石获得量', '经验值获得量'))]])

    def test_legacy_multiple_targets_show_first_and_explain_save_conversion(self):
        result = self.panel({'targets':['烧烤', '干货']})
        self.assertEqual(result['initial'], ['烧烤'])
        self.assertEqual(result['sent']['data']['targets'], ['烧烤'])
        self.assertIn('原设置包含多个类别', result['markup'])
        self.assertIn('保存后', result['markup'])

    def test_switching_single_selection_back_to_all_submits_empty_array(self):
        result = self.panel({'targets':['烧烤']}, '')
        self.assertEqual(result['initial'], ['烧烤'])
        self.assertTrue(result['dirty'])
        self.assertEqual(result['sent']['data']['targets'], [])

    def test_dropdown_is_disabled_while_running_busy_or_disconnected(self):
        self.assertEqual(self.panel({})['disabledStates'], [False, False, True, True, False])

    def test_other_features_have_no_target_controls_or_payload(self):
        for feature in ({'id':'mana'}, {'id':'parallel_skills', 'growth_points':False}, {'id':'equipment_upgrade'}):
            with self.subTest(feature=feature):
                result = self.panel(feature)
                self.assertEqual(result['values'], [])
                self.assertNotIn('targets', result['sent']['data'])
                if feature['id'] == 'parallel_skills':
                    self.assertIs(result['sent']['data']['growth_points'], False)


if __name__ == '__main__':
    unittest.main()
