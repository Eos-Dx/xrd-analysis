import queue

import matplotlib.pyplot as plt
import numpy as np


def _tm():
    from hardware.difra.gui.main_window_ext import technical_measurements as tm

    return tm


class TechnicalRealtimeMixin:
    def _toggle_realtime(self, checked: bool):
        if checked:
            self._log_technical_event("Starting real-time measurement display")
            self._start_realtime()
            self.rtBtn.setText("Stop RT")
        else:
            self._log_technical_event("Stopping real-time measurement display")
            self._stop_realtime()
            self.rtBtn.setText("Real-time")

    def _start_realtime(self):
        tm = _tm()
        exposure = float(self.integrationTimeSpin.value())
        self._rt_queue = queue.Queue()

        plt.ion()
        detector_aliases = list(self.detector_controller.keys())
        n_det = len(detector_aliases)
        self._rt_img = {}
        self._rt_last_frame = {}

        fig, axes = plt.subplots(1, n_det, figsize=(5 * n_det, 5))
        if n_det == 1:
            axes = [axes]

        for ax, alias in zip(axes, detector_aliases):
            size = getattr(self.detector_controller[alias], "size", (256, 256))
            self._rt_img[alias] = ax.imshow(np.zeros(size), origin="lower", interpolation="none")
            ax.set_title(alias)
        self._rt_fig = fig
        plt.show()

        self._plot_timer = tm.QTimer(self)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._rt_plot_tick)
        self._plot_timer.start()

        def callback(frames_dict):
            for alias, frame in frames_dict.items():
                self._rt_last_frame[alias] = frame
            self._rt_queue.put(True)

        for controller in self.detector_controller.values():
            controller.start_stream(callback=callback, exposure=exposure, interval=0.0, frames=1)

    def _rt_plot_tick(self):
        while True:
            try:
                _ = self._rt_queue.get_nowait()
            except queue.Empty:
                break

        for alias in self._rt_img:
            frame = self._rt_last_frame.get(alias)
            if frame is not None:
                self._rt_img[alias].set_data(frame)
                self._rt_img[alias].set_clim(frame.min(), frame.max())
        self._rt_fig.canvas.draw_idle()

    def _stop_realtime(self):
        for controller in self.detector_controller.values():
            controller.stop_stream()
        if hasattr(self, "_plot_timer"):
            self._plot_timer.stop()
            del self._plot_timer
        plt.close(self._rt_fig)
        del self._rt_queue
        del self._rt_last_frame
