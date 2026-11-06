"""Persistent game-day completion, with the user-confirmed 08:00 UTC+8 reset."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

BEIJING = timezone(timedelta(hours=8), 'Asia/Shanghai')
# 入口规则 → 功能(每周期只跑一次的入口)。
DAILY_ENTRIES = {'driftinggirl_daily_open': 'daily_reward',
                 'driftinggirl_guild_open': 'guild_shop',
                 # 邮箱:用户要求每天只触发一次,与每日奖励同一天周期(08:00 刷新)。
                 # 入口模板要求绿色 N 提示,所以"当天已领完"时即便画面上有邮箱图标也不会进流程;
                 # 一旦本周期标记完成,后续即使再出现新邮件提示也等到下一个 08:00。
                 'driftinggirl_mailbox_open': 'mailbox'}
# 功能的收尾规则 → 看到它就说明本轮流程走完了。
DAILY_RETURNS = {'daily_reward': 'driftinggirl_daily_return',
                 'pass_daily': 'driftinggirl_pass_daily',
                 'guild_daily': 'driftinggirl_guild_daily',
                 'guild_shop': 'driftinggirl_guild_return',
                 'mailbox': 'driftinggirl_mailbox_return'}


class DailySchedule:
    def __init__(self, root, serial, clock=None):
        self.path = Path(root) / 'daily_state.json'
        self.serial = serial
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.completed = {}
        self.require_bonus = False
        self.bonus_checked = {}
        self.warning = None
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if (not isinstance(saved, dict) or saved.get('version') != 1
                    or not isinstance(saved.get('completed'), dict)):
                raise ValueError('Invalid daily state')
            if saved.get('serial') == serial:
                for feature, entry in saved['completed'].items():
                    if feature not in DAILY_RETURNS:
                        continue
                    if not isinstance(entry, dict) or not isinstance(entry.get('period'), str):
                        raise ValueError('Invalid daily completion')
                    date.fromisoformat(entry['period'])
                    completed_at = datetime.fromisoformat(entry['completed_at'])
                    if completed_at.tzinfo is None:
                        raise ValueError('Daily completion needs a timezone')
                self.completed = {key: value for key, value in saved['completed'].items()
                                  if key in DAILY_RETURNS}
                bonus = saved.get('bonus_checked', {})
                if bonus:
                    date.fromisoformat(bonus['period'])
                    if datetime.fromisoformat(bonus['checked_at']).tzinfo is None:
                        raise ValueError('Bonus check needs a timezone')
                    self.bonus_checked = bonus
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError):
            self.warning = '每日完成记录无法读取，将重新核对游戏中的领取和限购状态。'

    def window(self):
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError('The daily schedule clock must include a timezone')
        now = now.astimezone(BEIJING)
        start = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
        return start.date().isoformat(), start + timedelta(days=1)

    def period(self):
        return self.window()[0]

    def is_complete(self, feature):
        free = self.completed.get(feature, {}).get('period') == self.period()
        return free and (feature != 'daily_reward' or not self.require_bonus
                         or self.bonus_checked.get('period') == self.period())

    def mark_complete(self, feature, started_period):
        # A workflow that straddles reset must be checked again in the new period.
        if feature not in DAILY_RETURNS or started_period != self.period():
            return False
        updated = deepcopy(self.completed)
        if updated.get(feature, {}).get('period') != started_period:
            updated[feature] = {'period': started_period,
                                'completed_at': self.clock().astimezone(BEIJING).isoformat()}
        saved = {'version': 1, 'serial': self.serial, 'completed': updated}
        bonus = self.bonus_checked
        if feature == 'daily_reward' and self.require_bonus:
            bonus = {'period': started_period,
                     'checked_at': self.clock().astimezone(BEIJING).isoformat()}
        if bonus:
            saved['bonus_checked'] = bonus
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
        self.completed = updated
        self.bonus_checked = bonus
        return True

    def status(self):
        period, following = self.window()
        return {feature: {'period': period, 'completed': self.is_complete(feature),
                           **({'not_before': datetime.fromisoformat(period).replace(hour=(18 if feature=='guild_daily' else 10),tzinfo=BEIJING).isoformat(),
                               'next_eligible': (following+timedelta(hours=(10 if feature=='guild_daily' else 2))).isoformat(),
                               'delay_hours': 10 if feature=='guild_daily' else 2}
                              if feature in ('pass_daily','guild_daily') else {}),
                           **({'free_completed': self.completed.get(feature, {}).get('period') == period,
                               'bonus_pending': self.require_bonus and self.bonus_checked.get('period') != period}
                              if feature == 'daily_reward' else {}),
                           'reset_hour': 8, 'timezone': 'Asia/Shanghai', 'next_reset': following.isoformat()}
                for feature in DAILY_RETURNS}

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

# Shared reliability layer; retain all other deployed methods.
def mark_complete(self, feature, started_period):
    # A workflow that straddles reset must be checked again in the new period.
    if feature not in DAILY_RETURNS or started_period != self.period():
        return False
    updated = deepcopy(self.completed)
    if updated.get(feature, {}).get('period') != started_period:
        updated[feature] = {'period': started_period,
                            'completed_at': self.clock().astimezone(BEIJING).isoformat()}
    saved = {'version': 1, 'serial': self.serial, 'completed': updated}
    bonus = self.bonus_checked
    if feature == 'daily_reward' and self.require_bonus:
        bonus = {'period': started_period,
                 'checked_at': self.clock().astimezone(BEIJING).isoformat()}
    if bonus:
        saved['bonus_checked'] = bonus
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
    self.completed = updated
    self.bonus_checked = bonus
    return True

DailySchedule.mark_complete = mark_complete



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
