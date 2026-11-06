"""Require native sale at route start without changing any other menu option."""
import hashlib
import time

import cv2
import numpy as np


ACTION = 'enable-auto-sale'


def preserved_key(sailing, frame):
    if not sailing.menu(frame):
        return None
    # Upgrade and both priority buttons remain immutable during the sale
    # change. Coupon digits and animated scenery are deliberately excluded.
    bits = b''.join((np.max(frame[y:Y,x:X],axis=2) < threshold).tobytes()
                   for (x,y,X,Y),threshold in (((557,741,595,781),110),
                       ((260,520,415,574),70),((434,520,594,574),70)))
    # Selected priority has a gold background and white lettering. Its dark
    # text mask alone can be empty, so bind the background color as well.
    for x,X in ((260,415),(434,594)):
        hsv = cv2.cvtColor(frame[520:574,x:X],cv2.COLOR_RGB2HSV)
        gold = ((hsv[:,:,0]>=10)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>100)&(hsv[:,:,2]>180))
        bits += bytes([int(float(gold.mean()) > .25)])
    return hashlib.sha256(bits).hexdigest()


def arm(owner, frame, settings):
    """Persist the reviewed OFF menu before reserving its one allowed click."""
    if not settings or owner.sailing.sale_selected(frame) is not False:
        return False
    guard = owner.journal.get('auto_sale_guard')
    if guard is None:
        guard = dict(targets=list(settings[0]), before_extras=settings[1],
                     preserved=preserved_key(owner.sailing,frame), verified=False)
        owner.journal['auto_sale_guard'] = guard
        owner._save()
    return valid_off(owner, frame, settings)


def valid_off(owner, frame, settings):
    guard = owner.journal.get('auto_sale_guard')
    return bool(guard and not guard.get('verified') and settings
                and owner.sailing.sale_selected(frame) is False
                and list(settings[0]) == guard['targets']
                and settings[1] == guard['before_extras']
                and preserved_key(owner.sailing,frame) == guard['preserved'])


def verified_on(owner, frame, settings, attempts):
    """Adopt only a positively seen ON box after our reserved OFF-to-ON input.

    Call after the controller's normal continuous menu confirmation. Returning
    True does not itself start a route or extend any workflow deadline.
    """
    if getattr(owner,'preserve_native_sale',False):
        return bool(settings and owner.sailing.sale_selected(frame) is not None)
    if not settings or owner.sailing.sale_selected(frame) is not True:
        return False
    guard = owner.journal.get('auto_sale_guard')
    if guard is None or guard.get('verified'):
        return True
    if (attempts.get(ACTION) != 1 or list(settings[0]) != guard['targets']
            or preserved_key(owner.sailing,frame) != guard['preserved']):
        return False
    guard.update(verified=True, after_extras=settings[1], verified_at=time.time())
    if 'native_extras' in owner.journal:
        owner.journal['native_extras'] = settings[1]
    owner._save()
    owner.event('航巡：已连续确认自动贩卖勾选，其他目标、优先顺序和自动强化保持不变')
    return True


