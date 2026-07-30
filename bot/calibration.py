from __future__ import annotations

import base64
from collections.abc import Callable

import cv2
import mss
import numpy as np
import tkinter as tk

from .models import Point, Region


class ScreenSelector:
    """Frozen-screen, mouse-driven region/point selector."""

    def __init__(
        self,
        root: tk.Tk,
        monitor_number: int,
        title: str,
        on_region: Callable[[Region, np.ndarray], None] | None = None,
        on_point: Callable[[Point], None] | None = None,
        highlight_region: Region | None = None,
    ) -> None:
        self.root = root
        self.on_region = on_region
        self.on_point = on_point
        self.start: tuple[int, int] | None = None
        self.rectangle: int | None = None

        root.withdraw()
        root.update_idletasks()
        with mss.mss() as capture:
            number = monitor_number if monitor_number < len(capture.monitors) else 1
            self.monitor = capture.monitors[number]
            raw = np.asarray(capture.grab(self.monitor))
        self.frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        ok, encoded = cv2.imencode(".png", self.frame)
        if not ok:
            root.deiconify()
            raise RuntimeError("Could not encode the screen capture")

        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.geometry(
            f"{self.monitor['width']}x{self.monitor['height']}"
            f"+{self.monitor['left']}+{self.monitor['top']}"
        )
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))
        self.canvas = tk.Canvas(
            self.window,
            width=self.monitor["width"],
            height=self.monitor["height"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.create_rectangle(8, 8, 620, 44, fill="black", outline="#00ff55")
        self.canvas.create_text(
            18,
            26,
            anchor="w",
            text=f"{title}    (Escape cancels)",
            fill="#00ff55",
            font=("", 13, "bold"),
        )
        if highlight_region:
            r = highlight_region
            self.canvas.create_rectangle(
                r.left,
                r.top,
                r.left + r.width,
                r.top + r.height,
                outline="#00ff55",
                width=5,
            )
        self.window.bind("<Escape>", lambda _event: self._finish())
        if on_point:
            self.canvas.bind("<Button-1>", self._point_clicked)
        elif on_region:
            self.canvas.bind("<ButtonPress-1>", self._drag_started)
            self.canvas.bind("<B1-Motion>", self._dragged)
            self.canvas.bind("<ButtonRelease-1>", self._drag_finished)
        else:
            self.canvas.bind("<Button-1>", lambda _event: self._finish())
        self.window.focus_force()

    def _drag_started(self, event: tk.Event) -> None:
        self.start = (event.x, event.y)
        self.rectangle = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ff55", width=3
        )

    def _dragged(self, event: tk.Event) -> None:
        if self.start and self.rectangle:
            self.canvas.coords(
                self.rectangle, self.start[0], self.start[1], event.x, event.y
            )

    def _drag_finished(self, event: tk.Event) -> None:
        if not self.start or not self.on_region:
            return
        x1, y1 = self.start
        left, top = min(x1, event.x), min(y1, event.y)
        right, bottom = max(x1, event.x), max(y1, event.y)
        if right - left < 4 or bottom - top < 4:
            return
        region = Region(left, top, right - left, bottom - top)
        crop = self.frame[top:bottom, left:right].copy()
        callback = self.on_region
        self._finish()
        callback(region, crop)

    def _point_clicked(self, event: tk.Event) -> None:
        if not self.on_point:
            return
        point = Point(
            event.x + self.monitor["left"], event.y + self.monitor["top"]
        )
        callback = self.on_point
        self._finish()
        callback(point)

    def _finish(self) -> None:
        self.window.destroy()
        self.root.deiconify()
        self.root.lift()
