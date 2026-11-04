# Adapted from LmeSzinc/AzurLaneAutoScript @ 77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21
# Copyright the ALAS contributors. SPDX-License-Identifier: GPL-3.0-only
from .button import Button
from .timer import Timer

class ModuleBase:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.interval_timer = {}

    def ensure_button(self, button):
        if not isinstance(button, Button):
            raise TypeError('This adapter accepts ALAS image Buttons only')
        return button

    def loop(self, skip_first=True, timeout=None, capture=None):
        """
        A syntactic sugar to start a state loop

        Args:
            skip_first (bool): Usually to be True to reuse the previous screenshot
            timeout (int | float | Timer): Seconds of timeout or a Timer object
            capture (callable): Optional fresh capture transport for this loop

        Yields:
            np.ndarray: screenshot

        Examples:
            # state machine that handle clicking until destination
            for _ in self.loop():
                if self.appear(...):
                    break
                if self.appear_then_click(...):
                    continue

        Examples:
            # state machine with timeout
            for _ in self.loop(timeout=2):
                if self.appear(...):
                    logger.info('Wait success')
                    break
            else:
                logger.warning('Wait timeout')
        """
        if timeout is not None:
            if isinstance(timeout, Timer):
                timeout.reset()
            else:
                timeout = Timer.from_seconds(timeout).start()

        while 1:
            if timeout is not None:
                if timeout.reached():
                    return

            if skip_first:
                skip_first = False
            else:
                if capture is None:
                    self.device.screenshot()
                else:
                    self.device.image = capture()

            try:
                yield self.device.image
            except AttributeError:
                if capture is None:
                    self.device.screenshot()
                else:
                    self.device.image = capture()
                yield self.device.image


    def appear(self, button, offset=0, interval=0, similarity=0.85, threshold=10):
        """
        Args:
            button (Button, Template, HierarchyButton, str):
            offset (bool, int):
            interval (int, float): interval between two active events.
            similarity (int, float): 0 to 1.
            threshold (int, float): 0 to 255 if not use offset, smaller means more similar

        Returns:
            bool:

        Examples:
            Image detection:
            ```
            self.device.screenshot()
            self.appear(Button(area=(...), color=(...), button=(...))
            self.appear(Template(file='...')
            ```

            Hierarchy detection (detect elements with xpath):
            ```
            self.device.dump_hierarchy()
            self.appear('//*[@resource-id="..."]')
            ```
        """
        button = self.ensure_button(button)
        self.device.stuck_record_add(button)

        if interval:
            if button.name in self.interval_timer:
                if self.interval_timer[button.name].limit != interval:
                    self.interval_timer[button.name] = Timer(interval)
            else:
                self.interval_timer[button.name] = Timer(interval)
            if not self.interval_timer[button.name].reached():
                return False

        if offset:
            if isinstance(offset, bool):
                offset = self.config.BUTTON_OFFSET
            appear = button.match(self.device.image, offset=offset, similarity=similarity)
        else:
            appear = button.appear_on(self.device.image, threshold=threshold)

        if appear and interval:
            self.interval_timer[button.name].reset()

        return appear


    def appear_then_click(self, button, screenshot=False, genre='items', offset=0, interval=0, similarity=0.85,
                          threshold=30):
        button = self.ensure_button(button)
        appear = self.appear(button, offset=offset, interval=interval, similarity=similarity, threshold=threshold)
        if appear:
            if screenshot:
                self.device.sleep(self.config.WAIT_BEFORE_SAVING_SCREEN_SHOT)
                self.device.screenshot()
                self.device.save_screenshot(genre=genre)
            self.device.click(button)
        return appear
