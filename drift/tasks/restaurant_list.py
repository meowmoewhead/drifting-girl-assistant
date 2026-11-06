"""Overlapping cookbook scans, stable-frame gates and unique dish coverage."""
import hashlib

import cv2
import numpy as np

EXPECTED = 9
DRAG_MS = 1500
LEFT, RIGHT, Y = 270, 450, 400
BACK_LIMIT = 5


def same_list(left, right):
    if left is None or right is None or left.shape != right.shape:
        return False
    if float(np.abs(left.astype(float)-right.astype(float)).mean()) > .8:
        return False
    score = float(cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0, 0])
    return bool(np.isfinite(score) and score >= .999)


def reset_settle(task):
    task._atlas_still = None


def settled(task, key, frame):
    """Moving frames cannot vote, visit cards, select dishes or prove an edge."""
    now = task.clock()
    previous = getattr(task, '_atlas_still', None)
    if previous is not None and frame is previous['frame']:
        return False
    if (previous is None or now-previous['last_at'] > 12
            or not same_list(key, previous['key'])):
        task._atlas_still = dict(key=key.copy(), frame=frame, since=now,
                                 last_at=now, frames=1)
        task._confirmed(None, frame)
        return False
    previous.update(frame=frame, last_at=now, frames=previous['frames']+1)
    return previous['frames'] >= 3 and now-previous['since'] >= 2


def groups(task, category=None):
    from drift.tasks.restaurant_upgrade import unpack
    result = []
    for card in task.journal.get('visited', []):
        if category is not None and card['category'] != category:
            continue
        art = unpack(card['art'])
        identity = card.get('identity')
        found = next((g for g in result if g['category'] == card['category']
            and ((identity in g['identities']) if identity and g['identities']
                 else task.visual.art_matches(art, g['art']))), None)
        if found is None:
            result.append(dict(category=card['category'], art=art,
                identities={identity} if identity else set(), level=card['level']))
        else:
            if identity:
                found['identities'].add(identity)
            found['level'] = max(found['level'], card['level'])
    return result


def count(task, category=None):
    return len(groups(task, category))


def exhausted(task):
    covered = groups(task, task.journal['category'])
    done = set(task.journal.get('exhausted_dishes', []))
    return (len(covered) == EXPECTED and all(g['level'] == 20
            or bool(g['identities'] & done) for g in covered))


def fresh_list(task, action, frame):
    from drift.tasks.restaurant_upgrade import unpack
    snapshot = action.get('list_key')
    return bool(snapshot and same_list(task.visual.list_key(frame),
                                      unpack(snapshot)[:, :, 0]))


def _scan(task):
    category = task.journal['category']
    scan = task.journal.get('atlas_scan')
    if not scan or scan.get('category') != category:
        scan = dict(category=category, direction='forward', rescan_used=False,
                    back_steps=0)
        task.journal['atlas_scan'] = scan
        task._save()
        reset_settle(task)
    return scan


def _incomplete(task, total):
    category = task.journal['category']
    reason = f'去重仅确认{total}/{EXPECTED}种餐品，有界回扫后仍缺项；本类别未完成'
    items = task.journal.setdefault('incomplete_categories', [])
    if not any(item['category'] == category for item in items):
        items.append(dict(category=category, checked=total, reason=reason))
        task.journal['skipped'].append(dict(category=category, reason=reason))
        task.event('餐品强化：'+reason)
        task._save()
    task.order_hold = reason
    task._set('category_next')


def step_list(task, frame):
    from drift.tasks.restaurant_upgrade import pack, unpack
    category = task.journal.get('category')
    if task.visual.category(frame) != category:
        reset_settle(task)
        task._confirmed(None, frame)
        return True
    scan = _scan(task)
    key = task.visual.list_key(frame)
    if not settled(task, key, frame):
        task.order_hold = '等待至少三张新图、两秒确认餐品列表静止'
        return True
    previous = task.journal.get('previous_list')
    unchanged = (previous is not None and same_list(key, unpack(previous)[:, :, 0]))
    if task.phase == 'align':
        if unchanged or task.journal['aligns'] >= 4:
            task.journal['previous_list'] = None
            task._set('list', 1200)
            return True
        if task._confirmed(('align', category), frame):
            if task._act('align:'+str(category), (LEFT, Y), frame,
                         swipe=[RIGHT, Y, DRAG_MS], list_key=pack(key)):
                task.journal.update(previous_list=pack(key), aligns=task.journal['aligns']+1)
                reset_settle(task)
                task._save()
        return True
    task.order_hold = None
    for card in task._ordered_cards(frame):
        if task._exhausted_card(card) or task._visited(card):
            continue
        if card['level'] is None:
            task.order_hold = '等待最左侧未完成餐品的等级读清，不越过它选择右侧餐品'
            task._confirmed(None, frame)
            return True
        if card['level'] == 20:
            task.journal['visited'].append(dict(category=category, art=pack(card['art']), level=20))
            task.journal.update(scrolls=0, previous_list=None)
            unchanged = False
            task._save()
            continue
        art = pack(card['art'])
        identity = hashlib.sha256(card['art'].tobytes()).hexdigest()[:16]
        if task._confirmed(('dish', category, identity, card['level']), frame):
            if task._act('dish:'+str(category)+':'+identity, card['point'], frame,
                         art=art, level=card['level'], list_key=pack(key)):
                task.journal.update(card=dict(art=art, level=card['level'], point=list(card['point'])),
                    feed_counts={}, blocked_ingredients=[], stall=0, scrolls=0, previous_list=None)
                task.event(f"餐品强化：从左到右选中餐品，位置 {card['point'][0]}，Lv{card['level']}；完成当前餐品后再处理下一个")
                task.journal.pop('dish_identity', None)
                task._set('detail', 300)
        return True
    if not task._confirmed(('list', category), frame):
        return True
    if task._category_exhausted():
        task._complete_category()
        return True
    total = count(task, category)
    if scan['direction'] == 'forward' and (unchanged or task.journal['scrolls'] >= 5):
        scan.update(direction='backward', rescan_used=True)
        task.journal['previous_list'] = None
        task._confirmed(None, frame)
        task._save()
        task.event(f'餐品强化：已到列表末端，但去重仅{total}/{EXPECTED}种，开始一次有界回扫')
        return True
    backward = scan['direction'] == 'backward'
    if backward and (unchanged or scan['back_steps'] >= BACK_LIMIT):
        _incomplete(task, total)
        return True
    kind = f'scroll:{category}:{len(task.journal["visited"])}'+(':back' if backward else '')
    start, end = (LEFT, RIGHT) if backward else (RIGHT, LEFT)
    if task._act(kind, (start, Y), frame, swipe=[end, Y, DRAG_MS], list_key=pack(key)):
        task.journal['previous_list'] = pack(key)
        if backward:
            scan['back_steps'] += 1
        else:
            task.journal['scrolls'] += 1
        reset_settle(task)
        task._save()
    return True
