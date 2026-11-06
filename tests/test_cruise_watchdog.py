"""Regression cases for bounded native cruise recovery and fresh contacts."""
import unittest
from types import SimpleNamespace

from drift.tasks.cruise_watchdog import CruiseWatchdog, CruiseView, fresh_cruise_input


class Harness:
    def __init__(self):
        self.now = 0.
        self.clicks, self.events = [], []
        self.route, self.menu = False, False
        self.known = True
        self.scene = (8, 39)
        self.signature = 'fixed-place'
        self.settings = ((True, False, True, True, True), 'extras')
        self.on_observe = lambda: None
        self.on_click = lambda: True
        self.watchdog = CruiseWatchdog(running=lambda f: f['route'],
            observe=self.observe, click=self.click, event=self.events.append, clock=lambda: self.now)

    def frame(self):
        return dict(scene=self.scene, signature=self.signature, menu=self.menu,
                    route=None if self.menu else self.route,
                    settings=self.settings if self.menu else None, known=self.known)

    def observe(self, frame):
        self.on_observe()
        return frame if frame['known'] else None

    def click(self, kind, x, y, label, expected):
        if not self.on_click() or not self.watchdog.input_allowed(kind):
            return False
        self.clicks.append(kind)
        return True

    def tick(self, seconds=1., times=1):
        result = None
        for _ in range(times):
            self.now += seconds
            result = self.watchdog.step(self.frame())
        return result

    def open(self):
        self.tick(times=91)
        assert self.clicks == ['cruise-open']

    def start(self):
        self.open()
        self.menu = True
        self.tick(times=4)
        assert self.clicks == ['cruise-open', 'cruise-start']


class CruiseWatchdogTests(unittest.TestCase):
    def test_running_route_never_acts(self):
        h = Harness(); h.route = True
        h.tick(times=300)
        self.assertEqual(h.clicks, [])

    def test_quiet_window_requires_continuous_new_frames(self):
        h = Harness()
        h.tick(times=90)
        self.assertEqual(h.clicks, [])
        h.tick()
        self.assertEqual(h.clicks, ['cruise-open'])

    def test_full_sequence_and_menu_ownership(self):
        h = Harness(); h.open()
        self.assertTrue(h.tick())
        h.menu = True
        for _ in range(4):
            self.assertTrue(h.tick())
        self.assertEqual(h.clicks, ['cruise-open', 'cruise-start'])
        h.menu, h.route = False, True
        h.tick(times=4)
        self.assertEqual(h.watchdog.state, 'idle')
        self.assertIn('航巡已恢复运行', h.watchdog.hold)
        h.tick(times=300)
        self.assertEqual(len(h.clicks), 2)

    def test_menu_bad_frames_reset_confirmation(self):
        h = Harness(); h.open()
        for visible in (True, False, True, False, True, False):
            h.menu = visible
            h.tick()
        self.assertEqual(h.clicks, ['cruise-open'])
        h.menu = True
        h.tick(times=3)
        self.assertEqual(h.clicks, ['cruise-open'])  # only 2 seconds
        h.tick()
        self.assertEqual(h.clicks[-1], 'cruise-start')

    def test_unknown_frames_reset_confirmation(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=2)
        h.known = False; h.tick()
        h.known = True; h.tick(times=3)
        self.assertEqual(h.clicks, ['cruise-open'])

    def test_repeated_object_cannot_be_three_frames(self):
        h = Harness(); h.open(); h.menu = True
        frame = h.frame()
        for _ in range(8):
            h.now += 1
            h.watchdog.step(frame)
        self.assertEqual(h.clicks, ['cruise-open'])
        self.assertEqual(h.watchdog._seen, 1)

    def test_menu_third_frame_after_deadline_cannot_click(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=2)
        h.now = h.watchdog.deadline
        h.watchdog.step(h.frame())
        self.assertEqual(h.clicks, ['cruise-open'])
        self.assertEqual(h.watchdog.failures, 1)

    def test_slow_recognition_cannot_cross_deadline(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=2)
        h.on_observe = lambda: setattr(h, 'now', h.watchdog.deadline + .01)
        h.tick()
        self.assertEqual(h.clicks, ['cruise-open'])
        self.assertEqual(h.watchdog.failures, 1)

    def test_preinput_deadline_rechecked(self):
        h = Harness(); h.open(); h.menu = True
        def expire():
            h.now = h.watchdog.deadline
            return True
        h.on_click = expire
        h.tick(times=4)
        self.assertEqual(h.clicks, ['cruise-open'])
        self.assertEqual(h.watchdog.failures, 1)

    def test_suspend_keeps_deadline_but_discards_frames(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=2)
        deadline = h.watchdog.deadline
        h.watchdog.suspend()
        h.tick(times=3)
        self.assertEqual(h.watchdog.deadline, deadline)
        self.assertEqual(h.clicks, ['cruise-open'])
        h.tick()
        self.assertEqual(h.clicks[-1], 'cruise-start')

    def test_long_unobserved_gap_restarts_quiet_period(self):
        h = Harness(); h.tick(times=89)
        h.tick(seconds=20)
        self.assertEqual(h.clicks, [])
        h.tick(times=89)
        self.assertEqual(h.clicks, [])
        h.tick()
        self.assertEqual(h.clicks, ['cruise-open'])

    def test_context_change_during_quiet_restarts_window(self):
        h = Harness(); h.tick(times=89)
        h.signature = 'other-place'
        h.tick(times=4)
        self.assertEqual(h.clicks, [])

    def test_scene_change_in_menu_exits_without_start(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=2)
        h.scene = (9, 39)
        h.tick()
        self.assertEqual(h.clicks, ['cruise-open'])
        self.assertFalse(h.watchdog.busy)

    def test_settings_change_requires_new_continuous_proof(self):
        h = Harness(); h.open(); h.menu = True
        h.tick(times=3)
        h.settings = ((True, False, True, False, True), 'new-priority')
        h.tick(times=3)
        self.assertEqual(h.clicks, ['cruise-open'])
        h.tick()
        self.assertEqual(h.clicks[-1], 'cruise-start')

    def test_start_confirmation_resets_on_nonrunning_frame(self):
        h = Harness(); h.start()
        h.menu, h.route = False, True
        h.tick(times=3)
        h.route = False; h.tick()
        h.route = True; h.tick(times=3)
        self.assertEqual(h.watchdog.state, 'starting')
        h.tick()
        self.assertEqual(h.watchdog.state, 'idle')

    def test_start_timeout_does_not_claim_recovery_or_retry(self):
        h = Harness(); h.start()
        h.tick(times=31)
        self.assertEqual(h.watchdog.failures, 1)
        self.assertIn('没有变成运行', h.watchdog.hold)
        h.menu = False
        h.tick(times=100)
        self.assertEqual(len(h.clicks), 2)

    def test_unknown_route_resets_quiet_period(self):
        h = Harness(); h.tick(times=89)
        h.route = None; h.tick()
        h.route = False; h.tick(times=4)
        self.assertEqual(h.clicks, [])


class FreshCruiseInputTests(unittest.TestCase):
    def invoke(self, *, capture_delay=0., validate_delay=0., allowed_until=10.,
               valid=True, receipt=True, stopped_at_capture=False):
        now, calls = [0.], []
        frame = object()
        def capture():
            calls.append('capture'); now[0] += capture_delay
            return frame
        def validate(f):
            self.assertIs(f, frame)
            calls.append('validate'); now[0] += validate_delay
            return valid
        def allowed():
            return now[0] < allowed_until and not (stopped_at_capture and 'capture' in calls)
        def touch(f, began):
            self.assertIs(f, frame)
            calls.append('touch')
            return {'transport': 'test'} if receipt else None
        result = fresh_cruise_input(capture=capture, validate=validate,
            allowed=allowed, touch=touch, clock=lambda: now[0])
        return result, calls

    def test_fresh_frame_precedes_single_contact(self):
        result, calls = self.invoke()
        self.assertIsNotNone(result)
        self.assertEqual(calls, ['capture', 'validate', 'touch'])

    def test_no_capture_or_contact_after_limit(self):
        result, calls = self.invoke(allowed_until=0.)
        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_stop_after_capture_prevents_contact(self):
        result, calls = self.invoke(stopped_at_capture=True)
        self.assertIsNone(result)
        self.assertEqual(calls, ['capture'])

    def test_changed_new_frame_prevents_contact(self):
        result, calls = self.invoke(valid=False)
        self.assertIsNone(result)
        self.assertNotIn('touch', calls)

    def test_deadline_during_capture_or_validation_prevents_contact(self):
        for kw in ({'capture_delay': .06}, {'validate_delay': .06}):
            with self.subTest(kw=kw):
                result, calls = self.invoke(allowed_until=.05, **kw)
                self.assertIsNone(result)
                self.assertNotIn('touch', calls)

    def test_stale_frame_refused_even_when_limits_remain(self):
        for kw in ({'capture_delay': .121}, {'validate_delay': .121}):
            with self.subTest(kw=kw):
                result, calls = self.invoke(**kw)
                self.assertIsNone(result)
                self.assertNotIn('touch', calls)

    def test_missing_receipt_never_retries_contact(self):
        result, calls = self.invoke(receipt=False)
        self.assertIsNone(result)
        self.assertEqual(calls.count('touch'), 1)

    def test_transport_cancellation_is_bounded_but_uncertain_down_is_fatal(self):
        for error in (InterruptedError('deadline before DOWN'), RuntimeError('uncertain DOWN')):
            contacts = []
            def touch(frame, began):
                contacts.append(began)
                raise error
            def invoke():
                return fresh_cruise_input(capture=object, validate=lambda f: True,
                    allowed=lambda: True, touch=touch)
            if isinstance(error, InterruptedError):
                self.assertIsNone(invoke())
            else:
                with self.assertRaises(RuntimeError):
                    invoke()
            self.assertEqual(len(contacts), 1)


class CruiseViewTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.frame = np.full((1280, 720, 3), 160, dtype=np.uint8)
        self.current = dict(rank=38, stage=9, menu=False, completion=False, route=False)
        self.options = ((True, False, True, True, True), 'extras')
        self.sale, self.parallel, self.off = True, True, True
        self.owner = SimpleNamespace(enabled=lambda: True, active=False,
            START_STAGE=9, HANDOFF_RANK=39, screen=lambda f: self.current,
            settings_key=lambda f: self.options, _parallel=lambda f: self.parallel,
            _off_icon=lambda f: self.off, sailing=SimpleNamespace(
                sale_selected=lambda f: self.sale, menu=lambda f: self.current['menu'],
                running=lambda f: self.current['route']))
        self.view = CruiseView(self.owner)
        self.owner.fishing=SimpleNamespace(_main_visible=lambda f:True)
        self.setup_values={'pearl':True,'material':False,'amber':True,'gem':True,'boss':True}
        self.owner.sailing.selected=lambda f,n:self.setup_values[n]
        self.owner.sailing.upgrade_selected=lambda f:True

    def test_handoff_ranks_defer_to_portal(self):
        for stage, rank in ((9, 39), (8, 40), (9, 40)):
            with self.subTest(stage=stage, rank=rank):
                self.current.update(stage=stage, rank=rank)
                self.assertIsNone(self.view.observe(self.frame))
        self.current.update(stage=8, rank=39)
        self.assertIsNotNone(self.view.observe(self.frame))

    def test_disabled_or_active_owner_cannot_resume(self):
        self.owner.active = True
        self.assertIsNone(self.view.observe(self.frame))
        self.owner.active = False
        self.owner.enabled = lambda: False
        self.assertIsNone(self.view.observe(self.frame))

    def test_changed_fixed_text_refused_on_new_capture(self):
        expected = self.view.observe(self.frame)
        self.assertTrue(self.view.validate('cruise-open', self.frame, expected))
        for box in ((12, 286), (45, 40), (9, 71)):
            with self.subTest(pixel=box):
                changed = self.frame.copy()
                changed[box] = (0, 250, 255) if box == (45, 40) else (0, 0, 0)
                self.assertFalse(self.view.validate('cruise-open', changed, expected))

    def test_off_world_or_running_arrow_refused(self):
        expected = self.view.observe(self.frame)
        self.parallel = False
        self.assertFalse(self.view.validate('cruise-open', self.frame, expected))
        self.parallel = True; self.off = False
        self.assertFalse(self.view.validate('cruise-open', self.frame, expected))

    def test_menu_preserves_options_and_requires_auto_sale(self):
        self.current.update(menu=True, route=None)
        expected = self.view.observe(self.frame)
        self.assertTrue(self.view.validate('cruise-start', self.frame, expected))
        self.options = ((False, False, True, True, True), 'other-extras')
        self.assertFalse(self.view.validate('cruise-start', self.frame, expected))
        self.options = expected['settings']; self.sale = False
        self.assertIsNone(self.view.observe(self.frame))
        self.assertFalse(self.view.validate('cruise-start', self.frame, expected))

    def test_setup_fresh_guard_requires_stopped_same_scene_and_requested_option(self):
        from drift.tasks.cruise_setup import DESIRED,OPEN_AT,START_AT,SALE_AT
        expected=self.view.observe_setup(self.frame)
        self.assertTrue(self.view.validate_setup(self.frame,expected,OPEN_AT,DESIRED))
        self.off=False
        self.assertFalse(self.view.validate_setup(self.frame,expected,OPEN_AT,DESIRED))
        self.off=True;self.current.update(menu=True,route=None)
        expected=self.view.observe_setup(self.frame)
        self.assertTrue(self.view.validate_setup(self.frame,expected,START_AT,DESIRED))
        self.sale=False
        self.assertFalse(self.view.validate_setup(self.frame,expected,START_AT,DESIRED))
        self.assertTrue(self.view.validate_setup(self.frame,expected,SALE_AT,DESIRED))
        changed=self.frame.copy();changed[12,286]=0
        self.assertFalse(self.view.validate_setup(changed,expected,SALE_AT,DESIRED))
        self.setup_values['boss']=None
        self.assertFalse(self.view.validate_setup(self.frame,expected,SALE_AT,DESIRED))

    def test_setup_never_takes_over_portal_handoff(self):
        for stage,rank in ((9,39),(8,40),(9,40)):
            self.current.update(stage=stage,rank=rank)
            self.assertIsNone(self.view.observe_setup(self.frame))


if __name__ == '__main__':
    unittest.main()
