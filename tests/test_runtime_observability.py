"""Diagnostics must explain refusals without granting inputs or changing journals."""
from copy import deepcopy
import io
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drift.core.observability import CandidateObservation
from drift.runtime import Clicker
from drift.tasks.cruise_watchdog import CruiseWatchdog, fresh_cruise_input
from drift.tasks.meter_calibration import MeterCalibration
from drift.web.server import make_handler


class StopObservationTests(unittest.TestCase):
    def build(self):
        app = object.__new__(Clicker)
        app.lock, app.stop_event = threading.RLock(), threading.Event()
        app.thread = SimpleNamespace(is_alive=lambda: True)
        app.state = dict(stage='game_recovery', mode='clicking', clicks=7, frame_id='unique-frame')
        app.runtime = SimpleNamespace(request_stop=lambda: self.assertTrue(app.stop_event.is_set()))
        events = []
        app.event = events.append
        return app, events

    def test_stop_signals_transport_before_observation_and_records_origin(self):
        app, events = self.build()
        app.stop(source='http_api', peer='127.0.0.1')
        row = app.state['last_stop_request']
        self.assertTrue(app.stop_event.is_set())
        self.assertEqual(row['source'], 'http_api')
        self.assertEqual(row['stage'], 'game_recovery')
        self.assertEqual(row['clicks'], 7)
        self.assertTrue(row['was_running'])
        self.assertFalse(row['already_requested'])
        self.assertLessEqual(len(row['callers']), 5)
        self.assertIn('test_stop_signals', row['callers'][0]['function'])
        self.assertEqual(set(row['callers'][0]), {'file', 'function', 'line'})
        self.assertEqual(app.state['mode'], 'stopping')
        self.assertEqual(events[-1], '已请求停止')
        app.stop()
        self.assertTrue(app.state['last_stop_request']['already_requested'])

    def test_api_origin_uses_socket_peer_and_does_not_log_request_data(self):
        app, events = self.build()
        with patch('drift.web.server.secrets.token_hex', return_value='private-session-token'):
            handler = object.__new__(make_handler(app))
        body = json.dumps(dict(source='forged-source', token='secret-body')).encode()
        handler.headers = {'X-Session-Token': 'private-session-token', 'Content-Length': str(len(body))}
        handler.path, handler.client_address = '/api/stop', ('127.0.0.1', 9876)
        handler.rfile, handler.respond = io.BytesIO(body), Mock()
        handler.do_POST()
        self.assertEqual(app.state['last_stop_request']['source'], 'http_api')
        text = json.dumps(events)
        for secret in ('forged-source', 'secret-body', 'private-session-token'):
            self.assertNotIn(secret, text)


class CandidateObservationTests(unittest.TestCase):
    def build(self):
        now, events = [0.], []
        trace = CandidateObservation(events.append, clock=lambda: now[0])
        trace.begin(12345., rank=38, stage=9, player=1500, value=1400, high=1600)
        return trace, now, events

    def test_summary_counts_context_and_runtime_blocks_with_rate_limit(self):
        trace, now, events = self.build()
        for second in range(60):
            now[0] = second
            trace.check('context_confirmation', key=(38, second//30))
        self.assertEqual(len(events), 1)
        now[0] = 60.
        trace.check('runtime_popup', detail='daily_reset')
        self.assertEqual(len(events), 2)
        now[0] = 90.
        trace.finish('worker_exit')
        row = trace.snapshot()
        self.assertEqual(row['checks'], 61)
        self.assertEqual(row['context_changes'], 1)
        self.assertEqual(row['blocking_transitions'], 1)
        self.assertEqual(row['elapsed_seconds'], 90.)
        self.assertEqual(row['outcome'], 'worker_exit')
        trace.finish('another_reason')
        trace.check('battle')
        self.assertEqual(len(events), 3)
        self.assertEqual(trace.snapshot(), row)
        row['reasons']['forged'] = 1
        self.assertNotIn('forged', trace.snapshot()['reasons'])

    def test_diagnostic_logger_failure_is_nonfatal_and_storage_is_bounded(self):
        trace, now, events = self.build()
        trace.event = Mock(side_effect=OSError('disk'))
        for n in range(200):
            now[0] += 1
            trace.check('reason_'+str(n), detail='x'*1000)
        self.assertLessEqual(len(trace.snapshot()['reasons']), 25)
        self.assertEqual(len(trace.snapshot()['detail']), 160)
        self.assertEqual(trace.snapshot()['logging_error'], 'OSError')

    def candidate(self):
        cal = object.__new__(MeterCalibration)
        cal.journal = dict(pending=False, deadline=55., attempts={'cancel': 1})
        cal.phase, cal.candidate_at, cal.interval = 'idle', None, 5400.
        cal.enabled = lambda: True
        cal._next_check = lambda: 0.
        cal._restoring_only = lambda: False
        cal._report_wait, cal._save = Mock(), Mock()
        cal.event = Mock()
        cal._confirmed = Mock(return_value=False)
        cal.screen = Mock(return_value=dict(rank=38, stage=9, route=True, menu=False))
        cal.fishing = SimpleNamespace(measure=Mock(return_value=dict(
            player=1500, value=1400, high=1600, context=('fixed',))))
        cal.needs_adjustment = lambda data: data is not None and data['player']-data['value'] > 50
        return cal

    def test_candidate_wait_block_and_exit_do_not_mutate_journal_or_limits(self):
        cal = self.candidate()
        before = deepcopy(cal.journal)
        self.assertFalse(cal._step(object()))
        started = cal.candidate_at
        cal.observe_candidate_blocked('game_recovery')
        cal.observe_candidate_blocked('popup', 'daily_reset')
        cal.finish_candidate_observation('worker_exit')
        self.assertEqual(cal.journal, before)
        self.assertEqual(cal.candidate_at, started)
        cal._save.assert_not_called()
        self.assertEqual(cal._candidate_trace.snapshot()['reasons']['context_confirmation'], 1)
        self.assertEqual(cal._candidate_trace.snapshot()['outcome'], 'worker_exit')

    def test_unconfirmed_context_does_not_add_extra_fishing_detection(self):
        cal = self.candidate()
        cal._candidate_ready = Mock(side_effect=AssertionError('must short circuit'))
        self.assertFalse(cal._step(object()))
        cal._candidate_ready.assert_not_called()

    def test_confirmed_battle_has_separate_reason_without_result_or_idle_scans(self):
        cal = self.candidate()
        cal._confirmed.return_value = True
        cal.fishing.real_battle = Mock(return_value=True)
        cal._catch_result = Mock(side_effect=AssertionError('must short circuit'))
        self.assertFalse(cal._step(object()))
        self.assertEqual(cal._candidate_trace.snapshot()['reasons'], {'battle': 1})
        cal._catch_result.assert_not_called()

    def test_gap_closed_finishes_observation_without_writing_a_journal(self):
        cal = self.candidate()
        cal._step(object())
        cal.fishing.measure.return_value['value'] = 1490
        self.assertFalse(cal._step(object()))
        self.assertIsNone(cal.candidate_at)
        self.assertEqual(cal._candidate_trace.snapshot()['outcome'], 'gap_closed')
        cal._save.assert_not_called()

    def test_candidate_timeout_keeps_original_failure_backoff_and_spent_inputs(self):
        cal = self.candidate()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000.):
            cal._step(object())
        cal._set = Mock()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1300.):
            self.assertFalse(cal._step(object()))
        self.assertEqual(cal.journal['next_check'], 1600.)
        self.assertEqual(cal.journal['attempts'], {'cancel': 1})
        self.assertEqual(cal.journal['deadline'], 55.)
        self.assertEqual(cal._candidate_trace.snapshot()['outcome'], 'candidate_timeout')
        cal._save.assert_called_once()

    def test_ready_candidate_starts_original_90_second_window_without_input(self):
        cal = self.candidate()
        cal._confirmed.return_value = True
        cal._candidate_ready = Mock(return_value=True)
        cal.world = SimpleNamespace(_stage_strokes=lambda f: b'fixed', training=SimpleNamespace(state={}))
        cal.fishing.cast_data = Mock(return_value=None)
        cal._set = Mock()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000.):
            self.assertTrue(cal._step(object()))
        self.assertEqual(cal.journal['deadline'], 1090.)
        self.assertTrue(cal.journal['restore_running'])
        self.assertEqual(cal.journal['route_policy'], 'preserve')
        self.assertEqual(cal.journal['attempts'], {})
        self.assertEqual(cal._candidate_trace.snapshot()['outcome'], 'execution_started')
        cal._save.assert_called_once()


class CruiseTimingObservationTests(unittest.TestCase):
    def test_rejection_records_capture_and_validation_separately_after_no_touch(self):
        now, rejected = [0.], []
        def capture():
            now[0] += .025
            return object()
        def validate(frame):
            now[0] += .100
            return True
        touch = Mock()
        result = fresh_cruise_input(capture=capture, validate=validate, allowed=lambda: True,
            touch=touch, clock=lambda: now[0], rejected=rejected.append)
        self.assertIsNone(result)
        touch.assert_not_called()
        self.assertEqual(rejected, [dict(reason='stale_frame', capture_ms=25.,
                                          validation_ms=100., elapsed_ms=125.)])

    def test_no_diagnostic_callback_before_successful_contact(self):
        calls = []
        result = fresh_cruise_input(capture=object, validate=lambda f: True, allowed=lambda: True,
            touch=lambda f, t: calls.append('touch') or {}, clock=lambda: 0.,
            rejected=lambda d: calls.append('diagnostic'))
        self.assertIsNotNone(result)
        self.assertEqual(calls, ['touch'])

    def test_diagnostic_exception_cannot_change_refusal_to_failure_or_touch(self):
        touch = Mock()
        self.assertIsNone(fresh_cruise_input(capture=object, validate=lambda f: False,
            allowed=lambda: True, touch=touch, rejected=Mock(side_effect=OSError('disk'))))
        touch.assert_not_called()

    def test_rejection_counts_throttled_without_moving_watchdog_deadline(self):
        now, events = [0.], []
        w = CruiseWatchdog(running=None, observe=None, click=None, event=events.append, clock=lambda: now[0])
        w.deadline, w.retry_at = 30., 600.
        for n in range(60):
            now[0] = n
            w.observe_input_rejection('cruise-start', dict(reason='stale_frame', capture_ms=25.,
                validation_ms=100., elapsed_ms=125.))
        self.assertEqual(len(events), 1)
        self.assertEqual(w.input_rejections['cruise-start:stale_frame'], 60)
        self.assertEqual((w.deadline, w.retry_at), (30., 600.))


if __name__ == '__main__':
    unittest.main()
