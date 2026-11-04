"""二维地图搜索的离线行为测试；所有输入和拖动后截图均由假回调提供。"""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np

from drift.tasks import restaurant_upgrade as module
from drift.tasks.restaurant_upgrade import RestaurantUpgrade


class RestaurantMapScanTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.now = 100
        self.points = {}
        self.current = self.frame(10)
        self.after = None
        self.deliver_after = True
        self.issued = []
        self.task = self.controller()
        self.task.phase = 'map'
        self.task.journal.update(pending=True, phase='map', destination='restaurant',
            scans=0, attempts={}, deadline=10000, phase_deadline=5000,
            origin=dict(parallel=False, route=True), resume_route=True)
        self.task._save()

    @staticmethod
    def frame(value):
        return np.full((1280, 720, 3), value, dtype=np.uint8)

    def controller(self):
        with patch.object(module, 'RestaurantVisual'), patch.object(module, 'GuildVisual'), \
             patch.object(module, 'StoneFarm'), patch.object(module, 'BackpackSale'):
            task = RestaurantUpgrade(self.root, None, Mock(), Mock(), 'offline',
                                     clock=lambda: self.now)
        task.same_world = Mock(return_value=True)
        task.map_ready = Mock(return_value=True)
        task.view._off_icon.return_value = True
        task._confirmed = Mock(return_value=True)
        task.visual.point.side_effect = lambda frame, name: (
            self.points.get(int(frame[200, 50, 0])) if name == 'map-restaurant' else None)

        def issue(request):
            # 与运行时一样，输入前重新匹配模板并预留额度，不构造任何设备。
            if not task.validate(request, self.current.copy()) or not task.reserve(request):
                return False
            self.issued.append(request)
            task.input_issued(request, dict(input_id='offline'))
            if request['kind'].startswith('scan:'):
                before = self.current.copy()
                self.current = (self.after.copy() if self.after is not None else
                                self.frame((int(before[200, 50, 0]) + 10) % 240))
                if self.deliver_after and hasattr(task, 'observe_map_scan'):
                    task.observe_map_scan(before, self.current)
            return True

        task.action = Mock(side_effect=issue)
        return task

    def step(self):
        self.now += 4
        return self.task.step(self.current.copy(), due=False)

    def scans(self):
        return [a for a in self.issued if a['kind'].startswith('scan:')]

    def directions(self):
        result = []
        for action in self.scans():
            x, y = action['x'], action['y']
            X, Y, ms = action['swipe']
            self.assertEqual(ms, module.SCAN_DRAG_MS)
            for px, py in ((x, y), (X, Y)):
                self.assertTrue(20 <= px <= 700 and 175 <= py <= 1115)
            self.assertTrue((x == X) != (y == Y))
            result.append(('east' if X < x else 'west') if y == Y else
                          ('south' if Y < y else 'north'))
        return result

    def test_grid_has_three_rows_eight_steps_and_alternating_directions(self):
        for _ in range(26):
            self.step()
        self.assertEqual(self.directions(), ['east'] * 8 + ['south'] +
                         ['west'] * 8 + ['south'] + ['east'] * 8)
        self.assertEqual(self.task.journal['scans'], 26)
        self.step()
        self.assertEqual(len(self.scans()), 26)
        self.assertEqual(self.task.phase, 'cleanup')

    def test_scan_input_limit_is_forty(self):
        self.assertEqual(RestaurantUpgrade.input_limit('scan:restaurant'), 40)
        self.assertEqual(RestaurantUpgrade.input_limit('scan:origin'), 40)
        self.assertEqual(getattr(module, 'SCAN_LIMIT', None), 40)

    def test_forty_scans_trigger_cleanup_without_another_input(self):
        self.task.journal.update(scans=40, attempts={'scan:restaurant': 40})
        self.task._save()
        self.step()
        self.assertEqual(self.task.phase, 'cleanup')
        self.assertIn('搜索已达上限', self.task.journal['failure']['reason'])
        self.task.action.assert_not_called()

    def test_post_drag_target_is_selected_in_same_step_and_stops_scanning(self):
        self.points[20] = (310, 420)
        self.step()
        self.assertEqual([a['kind'] for a in self.issued],
                         ['scan:restaurant', 'map-select:restaurant'])
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual((self.issued[-1]['x'], self.issued[-1]['y']), (310, 420))
        self.assertEqual(self.task.journal['scan_hint'], [['drag', 'east']])
        self.step()
        self.assertEqual(len(self.scans()), 1)

    def test_visible_target_prevents_all_scan_inputs(self):
        self.points[10] = (120, 350)
        self.step()
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(self.scans(), [])
        self.assertEqual(self.task.journal.get('scan_hint'), [])

    def test_hint_replays_first_and_then_falls_back_to_grid(self):
        self.task.journal['scan_hint'] = [['drag', 'west'], ['drag', 'north'], ['drag', 'east']]
        for _ in range(4):
            self.step()
        self.assertEqual(self.directions(), ['west', 'north', 'east', 'east'])
        self.assertEqual(self.task.journal['scans'], 4)
        self.assertEqual(self.task.journal['attempts']['scan:restaurant'], 4)
        events = [c.args[0] for c in self.task.event.call_args_list]
        self.assertEqual(sum('按上次成功的拖动路径重放（3 步）' in e for e in events), 1)

    def test_hint_hit_saves_actual_path_and_does_not_replay_remaining_steps(self):
        self.task.journal['scan_hint'] = [['drag', 'west'], ['drag', 'south'], ['drag', 'east']]
        self.points[30] = (260, 330)
        self.step()
        self.step()
        self.assertEqual(self.directions(), ['west', 'south'])
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(self.task.journal['scan_hint'], [['drag', 'west'], ['drag', 'south']])
        saved = json.loads(self.task.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['scan_hint'], self.task.journal['scan_hint'])

    def test_successful_hint_survives_finish_restart_and_new_round(self):
        self.points[20] = (300, 400)
        self.step()
        self.task._finish(True)
        self.task = self.controller()
        self.task.start_screen = Mock(return_value=dict(route=True))
        self.task.step(self.current.copy(), due=True)
        self.assertEqual(self.task.journal.get('scan_hint'), [['drag', 'east']])
        self.assertNotIn('scan_state', self.task.journal)

    def test_origin_navigation_never_replays_or_overwrites_restaurant_hint(self):
        hint = [['drag', 'west'], ['drag', 'south']]
        self.task.journal['scan_hint'] = hint.copy()
        self.task._nav('origin')
        self.task._set('map')
        self.task.target = Mock(return_value=None)
        self.step()
        self.assertEqual(self.directions(), ['east'])
        self.task.target.return_value = (200, 300)
        self.step()
        self.assertEqual(self.task.journal['scan_hint'], hint)

    def test_identical_map_frames_mark_edges_and_end_early(self):
        self.after = self.current.copy()
        for _ in range(8):
            if self.task.phase != 'map':
                break
            self.step()
        self.assertEqual(self.task.phase, 'cleanup')
        directions = self.directions()
        self.assertLessEqual(len(directions), 4)
        self.assertEqual(len(directions), len(set(directions)))
        self.assertTrue({'east', 'west', 'south'}.issubset(directions))
        self.assertIn('到头', self.task.journal['failure']['reason'])
        self.assertTrue(any('向东到头' in c.args[0] for c in self.task.event.call_args_list))

    def test_changes_outside_map_do_not_hide_boundary(self):
        self.after = self.current.copy()
        self.after[:175] = 220
        self.after[1115:] = 220
        self.step()
        self.after = None
        self.step()
        self.assertEqual(self.directions(), ['east', 'west'])

    def test_moving_map_does_not_report_boundary(self):
        for _ in range(4):
            self.step()
        self.assertEqual(self.directions(), ['east'] * 4)
        self.assertFalse(any('到头' in c.args[0] for c in self.task.event.call_args_list))

    def test_rejected_drag_does_not_advance_path_hint_or_scan_count(self):
        self.task.journal['scan_hint'] = [['drag', 'west'], ['drag', 'south']]
        action = self.task.action
        self.task.action = Mock(return_value=False)
        self.step()
        self.assertEqual(self.task.journal['scans'], 0)
        self.task.action = action
        self.step()
        self.assertEqual(self.directions(), ['west'])
        self.points[int(self.current[200, 50, 0])] = (200, 350)
        self.step()
        self.assertEqual(self.task.journal['scan_hint'], [['drag', 'west']])

    def test_failed_selection_keeps_previous_successful_hint(self):
        old_hint = [['drag', 'west'], ['drag', 'south']]
        self.task.journal['scan_hint'] = old_hint.copy()
        self.points[20] = (300, 400)
        action = self.task.action.side_effect
        self.task.action.side_effect = lambda a: False if a['kind'].startswith('map-select:') else action(a)
        self.step()
        self.assertEqual(self.task.journal['scan_hint'], old_hint)
        self.assertEqual(self.task.phase, 'map')
        self.assertEqual(self.directions(), ['west'])

    def test_pending_drag_checks_next_fresh_frame_without_another_scan(self):
        self.deliver_after = False
        self.points[20] = (300, 400)
        self.step()
        self.assertEqual(self.task.phase, 'map')
        self.task._confirmed = Mock(return_value=False)
        self.step()
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(self.directions(), ['east'])

    def test_hint_progress_and_boundary_survive_controller_restart(self):
        self.task.journal['scan_hint'] = [['drag', 'west'], ['drag', 'south']]
        self.step()
        self.task = self.controller()
        self.step()
        self.assertEqual(self.directions(), ['west', 'south'])
        self.assertEqual(self.task.journal['scans'], 2)

    def test_thirtieth_hint_drag_can_still_select_visible_target(self):
        self.task.journal['scan_hint'] = [['drag', 'west']] * 30
        for _ in range(29):
            self.step()
        self.after = self.frame(250)
        self.points[250] = (300, 400)
        self.step()
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(len(self.scans()), 30)
        self.assertEqual(self.task.journal['attempts']['scan:restaurant'], 30)
        self.assertEqual(self.task.journal['scan_hint'], [['drag', 'west']] * 30)

    def test_hint_and_fallback_share_forty_input_budget(self):
        self.task.journal['scan_hint'] = [['drag', 'west']] * 20
        for _ in range(45):
            if self.task.phase != 'map':
                break
            self.step()
        # 前 20 步是重放，其余由网格补齐；两者共用同一个额度。
        self.assertEqual(self.directions()[:20], ['west'] * 20)
        self.assertEqual(self.task.journal['scans'], module.SCAN_LIMIT)
        self.assertEqual(self.task.phase, 'cleanup')
        self.assertIn('搜索已达上限', self.task.journal['failure']['reason'])

    def test_hint_does_not_repeat_a_blocked_direction_in_same_row(self):
        self.task.journal['scan_hint'] = [['drag', 'east']] * 4 + [['drag', 'west']]
        self.after = self.current.copy()
        self.step()
        self.after = None
        self.step()
        self.assertEqual(self.directions(), ['east', 'west'])

    def test_boundary_is_rechecked_after_moving_to_another_row(self):
        self.after = self.current.copy()
        self.step()  # 东边界
        self.after = None
        for _ in range(9):
            self.step()
        dirs = self.directions()
        self.assertEqual(dirs[0], 'east')
        self.assertIn('south', dirs)
        first_row = dirs[1:dirs.index('south')]
        self.assertTrue(first_row)
        # 同一行内不再重复已证实不动的方向；换行后才允许重新试探。
        self.assertNotIn('east', first_row)

    def test_disappearing_target_is_rejected_by_fresh_input_validation(self):
        self.points[20] = (300, 400)
        issue = self.task.action.side_effect

        def disappearing(request):
            if request['kind'].startswith('map-select:'):
                self.current = self.frame(50)
            return issue(request)

        self.task.action.side_effect = disappearing
        self.step()
        self.assertEqual([a['kind'] for a in self.issued], ['scan:restaurant'])
        self.assertEqual(self.task.phase, 'map')
        self.assertNotIn('scan_hint', self.task.journal)

    def test_runtime_callback_returns_real_after_frame_without_device_io(self):
        # 单独执行生产回调的 AST；设备、锁、截图、证据均为离线替身。
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'drift/runtime.py').read_text(encoding='utf-8-sig'))
        callback = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == 'restaurant_click')
        device = SimpleNamespace(io_lock=nullcontext(), command=Mock())
        app = SimpleNamespace(device=device, stop_event=Mock(), lock=nullcontext(),
            state={'clicks': 0}, remember=Mock(return_value='offline-frame'),
            event=Mock(), restaurant_evidence=Mock())
        app.stop_event.is_set.return_value = False
        before, after = self.current.copy(), self.frame(20)
        runtime = SimpleNamespace(capture=Mock(side_effect=[before, after, after.copy(), after.copy()]))
        scope = dict(self=app, restaurant=self.task, runtime=runtime, max_clicks=None,
                     st=SimpleNamespace(run_deadline=None), uuid=uuid,
                     _time=lambda: SimpleNamespace(time=lambda: self.now))
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<offline restaurant_click>', 'exec'), scope)
        self.task.action = scope['restaurant_click']
        self.points[20] = (300, 400)
        self.step()
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(device.command.call_count, 2)
        self.assertEqual(self.task.journal['attempts'], {'scan:restaurant': 1, 'map-select:restaurant': 1})
        self.assertEqual(self.task.journal['scan_hint'], [['drag', 'east']])


if __name__ == '__main__':
    unittest.main()
