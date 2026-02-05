from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, InstructionGroup, Ellipse
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from kivy.uix.behaviors import DragBehavior
from kivy.uix.scatter import Scatter
import platform
import numpy as np
from kivy.clock import Clock
import threading
import time
try:
    import cv2
except Exception:
    cv2 = None
try:
    import mss
except Exception:
    mss = None
OPENCV_AVAILABLE = cv2 is not None
if not OPENCV_AVAILABLE:
    print("OpenCV not found, running in manual mode")
import sys
import os


class OverlayWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_point = None
        self.last_start_point = None
        self.last_vdir = (1.0, 0.0)
        self.path_group = InstructionGroup()
        self.canvas.add(self.path_group)
        self.debug_group = InstructionGroup()
        self.canvas.add(self.debug_group)
        Window.clearcolor = (0, 0, 0, 0)
        self.is_on = True
        self.line_width = 2.0
        self.line_color = (0.2, 0.8, 1.0, 0.95)
        self.max_path_length = 1200.0
        self._clickthrough = True
        self.auto_mode = False
        self._auto_ev = None
        self._detector = None
        self._detected_result = None
        self.ghost_radius = 12.0
        self.virtual_cue = None
        self.menu = ModMenu(
            on_toggle=self.set_enabled,
            on_reset=self.clear_path,
            on_thickness=self.set_thickness,
            on_color=self.set_color,
            on_clickthrough=self.set_click_through,
            on_length=self.set_length,
            on_auto=self.set_auto_mode,
        )
        self.add_widget(self.menu)
        self.menu.pos = (Window.width - 160, 80)
        self.menu.set_enabled(self.is_on)
        self.set_click_through(True)

    def on_touch_down(self, touch):
        if hasattr(self, "menu") and self.menu.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        self.start_point = (touch.x, touch.y)
        self.last_start_point = self.start_point
        self.update_path(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        if self.start_point is None:
            return False
        if hasattr(self, "menu") and self.menu.collide_point(*touch.pos):
            return super().on_touch_move(touch)
        self.update_path(touch.x, touch.y)
        sx, sy = self.start_point
        dx, dy = touch.x - sx, touch.y - sy
        m = float(np.hypot(dx, dy))
        if m > 0:
            self.last_vdir = (dx / m, dy / m)
        self.last_start_point = self.start_point
        return True

    def on_touch_up(self, touch):
        return True

    def set_enabled(self, value):
        self.is_on = bool(value)
        if hasattr(self, "menu"):
            self.menu.set_enabled(self.is_on)
        if not self.is_on:
            self.path_group.clear()
        else:
            if self.start_point is not None:
                self.update_path(Window.mouse_pos[0], Window.mouse_pos[1])

    def toggle_state(self, *_):
        self.is_on = not self.is_on
        if hasattr(self, "menu"):
            self.menu.set_enabled(self.is_on)
        if not self.is_on:
            self.path_group.clear()
        else:
            if self.start_point is not None:
                self.update_path(Window.mouse_pos[0], Window.mouse_pos[1])

    def clear_path(self, *_):
        self.start_point = None
        self.path_group.clear()

    def set_thickness(self, value):
        try:
            self.line_width = float(value)
        except Exception:
            self.line_width = 2.0
        if self.is_on and self.start_point is not None:
            self.update_path(Window.mouse_pos[0], Window.mouse_pos[1])

    def set_color(self, rgba):
        self.line_color = rgba
        if self.is_on and self.start_point is not None:
            self.update_path(Window.mouse_pos[0], Window.mouse_pos[1])

    def set_length(self, value):
        try:
            self.max_path_length = float(value)
        except Exception:
            self.max_path_length = 1200.0
        if self.is_on and self.start_point is not None:
            self.update_path(Window.mouse_pos[0], Window.mouse_pos[1])

    def set_click_through(self, enabled):
        self._clickthrough = bool(enabled)
        if platform.system().lower().startswith("win"):
            try:
                import win32gui, win32con
                if not Window.title:
                    Window.title = "PoolHelperApp Overlay"
                hwnd = win32gui.FindWindow(None, Window.title)
                if hwnd:
                    exstyle = win32gui.GetWindowLong(hwnd, -20)
                    if enabled:
                        exstyle |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
                    else:
                        exstyle &= ~win32con.WS_EX_TRANSPARENT
                        exstyle |= win32con.WS_EX_LAYERED
                    win32gui.SetWindowLong(hwnd, -20, exstyle)
                    win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, 2)
            except Exception:
                pass

    def set_auto_mode(self, enabled):
        self.auto_mode = bool(enabled)
        if self.auto_mode:
            if self._detector is None and cv2 is not None and mss is not None:
                self._detector = DetectorThread(self)
                self._detector.start()
            if self._auto_ev is None:
                self._auto_ev = Clock.schedule_interval(self._auto_tick, 0.033)
        else:
            if self._detector is not None:
                self._detector.stop()
                self._detector = None
            if self._auto_ev is not None:
                self._auto_ev.cancel()
                self._auto_ev = None

    def update_path(self, x, y):
        if self.start_point is None or not self.is_on:
            return
        sx, sy = self.start_point
        dx, dy = x - sx, y - sy
        if dx == 0 and dy == 0:
            return
        vdir = (dx, dy)
        m = float(np.hypot(dx, dy))
        if m > 0:
            self.last_vdir = (dx / m, dy / m)
        self.last_start_point = (sx, sy)
        if getattr(self, "balls", None):
            self.draw_paths_with_balls(sx, sy, vdir)
        else:
            self.update_path_with_dir(sx, sy, vdir)

    def update_path_with_dir(self, sx, sy, vdir):
        self.path_group.clear()
        self.debug_group.clear()
        w, h = Window.width, Window.height
        if not self.is_on:
            return
        v = np.array(vdir, dtype=float)
        m = np.linalg.norm(v)
        if m == 0:
            return
        v /= m
        pos = np.array([sx, sy], dtype=float)
        segments_drawn = 0
        while segments_drawn < 6:
            tx = np.inf if v[0] == 0 else ((w - pos[0]) / v[0] if v[0] > 0 else (0 - pos[0]) / v[0])
            ty = np.inf if v[1] == 0 else ((h - pos[1]) / v[1] if v[1] > 0 else (0 - pos[1]) / v[1])
            t = min(tx, ty)
            if not np.isfinite(t) or t <= 0:
                break
            hit = pos + t * v
            draw_alpha = min(self.line_color[3], 0.75)
            self.path_group.add(Color(self.line_color[0], self.line_color[1], self.line_color[2], draw_alpha))
            self.path_group.add(Line(points=[pos[0], pos[1], hit[0], hit[1]], width=self.line_width, dash_length=12))
            self.path_group.add(Color(self.line_color[0], self.line_color[1], self.line_color[2], draw_alpha))
            self.path_group.add(Line(circle=(hit[0], hit[1], self.ghost_radius), width=1.4))
            eps = 1e-7
            corner = abs(tx - ty) < eps
            if corner:
                v[0] *= -1
                v[1] *= -1
            else:
                if tx < ty:
                    v[0] *= -1
                else:
                    v[1] *= -1
            pos = hit + v * eps
            segments_drawn += 1

    def draw_paths_with_balls(self, sx, sy, vdir):
        self.path_group.clear()
        w, h = Window.width, Window.height
        cue_color = (1.0, 1.0, 1.0, 0.7)
        target_color = (0.2, 0.6, 1.0, 0.7)
        v = np.array(vdir, dtype=float)
        m = np.linalg.norm(v)
        if m == 0:
            return
        v /= m
        pos = np.array([sx, sy], dtype=float)
        cue_r = float(getattr(self, "cue_radius", self.ghost_radius))
        pockets = [
            (0.0, 0.0, 20.0),
            (w / 2.0, 0.0, 20.0),
            (w, 0.0, 20.0),
            (0.0, h, 20.0),
            (w / 2.0, h, 20.0),
            (w, h, 20.0),
        ]
        segments_drawn = 0
        while segments_drawn < 6:
            tx = np.inf if v[0] == 0 else ((w - pos[0]) / v[0] if v[0] > 0 else (0 - pos[0]) / v[0])
            ty = np.inf if v[1] == 0 else ((h - pos[1]) / v[1] if v[1] > 0 else (0 - pos[1]) / v[1])
            t_wall = min(tx, ty)
            t_ball = np.inf
            hit_ball = None
            for (bx, by, br) in getattr(self, "balls", []):
                c = np.array([bx, by], dtype=float)
                r = float(br + cue_r)
                oc = pos - c
                b = 2.0 * np.dot(v, oc)
                c2 = np.dot(oc, oc) - r * r
                disc = b * b - 4.0 * c2
                if disc <= 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                t_hit = None
                for tt in (t1, t2):
                    if tt > 1e-6:
                        t_hit = tt if t_hit is None else min(t_hit, tt)
                if t_hit is not None and t_hit < t_ball and t_hit < t_wall:
                    t_ball = t_hit
                    hit_ball = (bx, by, br)
            t_pocket = np.inf
            hit_pocket = None
            for (px, py, pr) in pockets:
                c = np.array([px, py], dtype=float)
                r = float(pr)
                oc = pos - c
                b = 2.0 * np.dot(v, oc)
                c2 = np.dot(oc, oc) - r * r
                disc = b * b - 4.0 * c2
                if disc <= 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                t_hit = None
                for tt in (t1, t2):
                    if tt > 1e-6:
                        t_hit = tt if t_hit is None else min(t_hit, tt)
                if t_hit is not None and t_hit < t_pocket and t_hit < t_ball and t_hit < t_wall:
                    t_pocket = t_hit
                    hit_pocket = (px, py, pr)
            if hit_pocket is not None:
                hit = pos + t_pocket * v
                self.path_group.add(Color(*cue_color))
                self.path_group.add(Line(points=[pos[0], pos[1], hit[0], hit[1]], width=self.line_width, dash_length=12))
                self.path_group.add(Color(0.9, 0.9, 0.2, 0.9))
                self.path_group.add(Line(circle=(hit_pocket[0], hit_pocket[1], hit_pocket[2]), width=2.0))
                break
            elif hit_ball is not None:
                hit = pos + t_ball * v
                self.path_group.add(Color(*cue_color))
                self.path_group.add(Line(points=[pos[0], pos[1], hit[0], hit[1]], width=self.line_width, dash_length=12))
                self.path_group.add(Color(*cue_color))
                self.path_group.add(Line(circle=(hit[0], hit[1], self.ghost_radius), width=1.6))
                bx, by, br = hit_ball
                n = np.array([bx, by], dtype=float) - hit
                nm = np.linalg.norm(n)
                if nm == 0:
                    n = v.copy()
                    nm = 1.0
                n /= nm
                v_target = n
                v_cue = v - n * np.dot(v, n)
                self._draw_ball_path((bx, by), v_target, target_color, 6, w, h)
                v = v_cue
                pos = hit + v * 1e-7
                segments_drawn += 1
            else:
                hit = pos + t_wall * v
                self.path_group.add(Color(*cue_color))
                self.path_group.add(Line(points=[pos[0], pos[1], hit[0], hit[1]], width=self.line_width, dash_length=12))
                self.path_group.add(Color(*cue_color))
                self.path_group.add(Line(circle=(hit[0], hit[1], self.ghost_radius), width=1.4))
                if tx < ty:
                    v[0] *= -1
                else:
                    v[1] *= -1
                pos = hit + v * 1e-7
                segments_drawn += 1

    def _draw_ball_path(self, start_xy, vdir, color_rgba, max_segs, w, h):
        pos = np.array([float(start_xy[0]), float(start_xy[1])], dtype=float)
        v = np.array(vdir, dtype=float)
        m = np.linalg.norm(v)
        if m == 0:
            return
        v /= m
        segs = 0
        while segs < max_segs:
            tx = np.inf if v[0] == 0 else ((w - pos[0]) / v[0] if v[0] > 0 else (0 - pos[0]) / v[0])
            ty = np.inf if v[1] == 0 else ((h - pos[1]) / v[1] if v[1] > 0 else (0 - pos[1]) / v[1])
            t = min(tx, ty)
            if not np.isfinite(t) or t <= 0:
                pos = pos + v * 1e-6
                continue
            hit = pos + t * v
            self.path_group.add(Color(*color_rgba))
            self.path_group.add(Line(points=[pos[0], pos[1], hit[0], hit[1]], width=max(1.0, self.line_width - 0.5), dash_length=12))
            eps = 1e-7
            if tx < ty:
                v[0] *= -1
            else:
                v[1] *= -1
            pos = hit + v * eps
            segs += 1
    def _auto_tick(self, dt):
        if not self.is_on:
            return
        res = self._detected_result
        if not res:
            cx = Window.width / 2.0
            cy = Window.height / 2.0
            if self.last_start_point is not None:
                cx, cy = self.last_start_point
            self.start_point = (cx, cy)
            self.balls = []
            self.cue_radius = self.ghost_radius
            self._set_virtual_cue_visible(True, center=(cx, cy))
            self.update_path_with_dir(cx, cy, self.last_vdir)
            return
        if len(res) == 4:
            cx, cy, radius, vdir = res
            balls_list = []
            stick_found = False
        elif len(res) == 5:
            cx, cy, radius, vdir, balls_list = res
            stick_found = False
        else:
            cx, cy, radius, vdir, balls_list, stick_found = res
        kx = float(cx)
        ky = float(Window.height - cy)
        self.start_point = (kx, ky)
        self.last_start_point = self.start_point
        self.ghost_radius = float(radius) if radius else 12.0
        self.cue_radius = self.ghost_radius
        self.balls = [(float(bx), float(Window.height - by), float(br)) for (bx, by, br) in balls_list]
        self.path_group.add(Color(1.0, 0.0, 0.0, 0.9))
        from kivy.graphics import Rectangle
        self.path_group.add(Rectangle(pos=(kx - 6, ky - 6), size=(12, 12)))
        vdir_kivy = (float(vdir[0]), float(-vdir[1]))
        mv = float(np.hypot(vdir_kivy[0], vdir_kivy[1]))
        if mv > 0:
            self.last_vdir = (vdir_kivy[0] / mv, vdir_kivy[1] / mv)
        self._set_virtual_cue_visible(len(self.balls) == 0, center=self.start_point)
        if self.balls:
            if not stick_found:
                nb = min(self.balls, key=lambda b: (b[0] - self.start_point[0]) ** 2 + (b[1] - self.start_point[1]) ** 2)
                aim = (nb[0] - self.start_point[0], nb[1] - self.start_point[1])
                am = float(np.hypot(aim[0], aim[1]))
                if am > 0:
                    vdir_kivy = (aim[0] / am, aim[1] / am)
            self.draw_paths_with_balls(self.start_point[0], self.start_point[1], vdir_kivy)
        else:
            self.update_path_with_dir(self.start_point[0], self.start_point[1], vdir_kivy)

    def _set_virtual_cue_visible(self, visible, center=None):
        if visible:
            if self.virtual_cue is None:
                self.virtual_cue = DraggableCircle(radius=int(self.ghost_radius), color_on=(1, 1, 1, 0.9), color_off=(1, 1, 1, 0.9), text_on="", text_off="", on_release=self._on_virtual_release)
                self.add_widget(self.virtual_cue)
            if center is not None:
                self.virtual_cue.center = center
        else:
            if self.virtual_cue is not None:
                try:
                    self.remove_widget(self.virtual_cue)
                except Exception:
                    pass
                self.virtual_cue = None

    def _on_virtual_release(self):
        if self.virtual_cue is None:
            return
        cx, cy = self.virtual_cue.center
        self.start_point = (cx, cy)
        self.last_start_point = self.start_point
        self.update_path_with_dir(cx, cy, self.last_vdir)

class ModMenu(FloatLayout):
    def __init__(self, on_toggle, on_reset, on_thickness, on_color, on_clickthrough, on_length, on_auto, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.icon_size = (48, 48)
        self.panel_size = (260, 190)
        self.expanded = False
        self._is_updating = False
        self._layout_scheduled = False
        self.on_toggle = on_toggle
        self.on_reset = on_reset
        self.on_thickness = on_thickness
        self.on_color = on_color
        self.on_clickthrough = on_clickthrough
        self.on_length = on_length
        self.on_auto = on_auto
        self.icon_scatter = Scatter(size_hint=(None, None), size=self.icon_size, do_rotation=False, do_scale=False)
        self.icon_btn = Button(text="≡", size_hint=(None, None), size=self.icon_size, background_normal="", background_color=(0.15, 0.15, 0.15, 0.85), color=(1, 1, 1, 1))
        self.icon_scatter.add_widget(self.icon_btn)
        self.add_widget(self.icon_scatter)
        self.icon_btn.bind(on_release=self._toggle_panel)
        self.panel = FloatLayout(size_hint=(None, None), size=self.panel_size, opacity=0)
        self.add_widget(self.panel)
        self._build_panel_content()
        self.bind(pos=self._layout, size=self._layout)
        self.icon_scatter.bind(pos=self._on_scatter_move)
        self._layout()

    def set_enabled(self, is_on):
        self.switch.active = bool(is_on)
        self.enable_label.text = "Enable Lines: ON" if is_on else "Enable Lines: OFF"

    def _toggle_panel(self, *_):
        self.expanded = not self.expanded
        self.panel.opacity = 1 if self.expanded else 0
        self.panel.disabled = not self.expanded
        self._layout()
        self.on_clickthrough(not self.expanded)

    def _layout(self, *_):
        self._is_updating = True
        base_x, base_y = self.icon_scatter.pos
        self.pos = (base_x, base_y)
        self.panel.pos = (base_x + self.icon_size[0] + 8, base_y)
        self.size = (
            self.icon_size[0] + (self.panel_size[0] + 8 if self.expanded else 0),
            max(self.icon_size[1], self.panel_size[1] if self.expanded else self.icon_size[1]),
        )
        self._refresh_panel_bg()
        self._is_updating = False
        self._layout_scheduled = False

    def _on_scatter_move(self, *_):
        if self._is_updating:
            return
        self.pos = self.icon_scatter.pos
        if not self._layout_scheduled:
            self._layout_scheduled = True
            from kivy.clock import Clock as _Clock
            _Clock.schedule_once(self._layout, 0)

    def _refresh_panel_bg(self):
        self.panel.canvas.clear()
        if not self.expanded:
            return
        from kivy.graphics import Rectangle
        with self.panel.canvas:
            Color(0.06, 0.06, 0.08, 0.85)
            Rectangle(pos=self.panel.pos, size=self.panel.size)

    def _build_panel_content(self):
        vbox = BoxLayout(orientation="vertical", size_hint=(None, None), size=(self.panel_size[0], self.panel_size[1]), pos=self.panel.pos, padding=10, spacing=8)
        # Switch row
        row1 = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=8)
        self.enable_label = Label(text="Enable Lines: ON", color=(1, 1, 1, 1))
        self.switch = Switch(active=True)
        self.switch.bind(active=lambda _, val: self._on_toggle(val))
        row1.add_widget(self.enable_label)
        # Length slider
        row2 = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=8)
        lbl_len = Label(text="Line Length", color=(1, 1, 1, 1))
        self.len_slider = Slider(min=200, max=3000, value=1200, step=50)
        self.len_slider.bind(value=lambda _, val: self.on_length(val))
        row2.add_widget(lbl_len)
        # Color buttons
        row3 = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=8)
        colors = [
            (0.2, 0.8, 1.0, 0.95),
            (1.0, 0.5, 0.2, 0.95),
            (0.6, 1.0, 0.3, 0.95),
            (1.0, 0.2, 0.7, 0.95),
            (1.0, 1.0, 0.2, 0.95),
        ]
        for c in colors:
            btn = Button(size_hint=(None, None), size=(36, 36), background_normal="")
            btn.background_color = c
            btn.text = ""
            btn.bind(on_release=lambda _, col=c: self.on_color(col))
        # Toggle and Reset
        toggle_btn = Button(text="Toggle", size_hint=(1, None), height=36, background_normal="", background_color=(0.2, 0.8, 0.2, 0.95), color=(0, 0, 0, 1))
        toggle_btn.bind(on_release=lambda *_: self._toggle_enabled())
        reset_btn = Button(text="Reset", size_hint=(1, None), height=36, background_normal="", background_color=(0.25, 0.25, 0.25, 0.9), color=(1, 1, 1, 1))
        row4 = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40, spacing=8)
        auto_lbl = Label(text="Auto Mode", color=(1, 1, 1, 1))
        self.auto_switch = Switch(active=False)
        self.auto_switch.bind(active=lambda _, val: self.on_auto(val))
        row4.add_widget(auto_lbl)
        row4.add_widget(self.auto_switch)
        reset_btn.bind(on_release=lambda *_: self.on_reset())
        vbox.add_widget(row1)
        vbox.add_widget(row2)
        vbox.add_widget(row4)
        vbox.add_widget(row3)
        vbox.add_widget(toggle_btn)
        vbox.add_widget(reset_btn)
        self.panel.add_widget(vbox)

    def _on_toggle(self, val):
        self.enable_label.text = "Enable Lines: ON" if val else "Enable Lines: OFF"
        self.on_toggle(val)

    def _toggle_enabled(self):
        self.switch.active = not self.switch.active
        self._on_toggle(self.switch.active)

class DetectorThread(threading.Thread):
    def __init__(self, overlay):
        super().__init__(daemon=True)
        self.overlay = overlay
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if cv2 is None or mss is None:
            return
        with mss.mss() as sct:
            while not self._stop:
                try:
                    mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    shot = sct.grab(mon)
                    frame = np.array(shot)[:, :, :3]
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40], dtype=np.uint8)
                    upper_green = np.array([85, 255, 255], dtype=np.uint8)
                    mask_green = cv2.inRange(hsv, lower_green, upper_green)
                    kernel = np.ones((5, 5), np.uint8)
                    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
                    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
                    cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        largest = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(largest) > 10000:
                            hull = cv2.convexHull(largest)
                            poly_mask = np.zeros_like(mask_green)
                            cv2.fillConvexPoly(poly_mask, hull, 255)
                            mask_green = poly_mask
                    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray0 = cv2.medianBlur(gray0, 5)
                    gray_masked = cv2.bitwise_and(gray0, gray0, mask=mask_green)
                    circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=15, minRadius=6, maxRadius=40)
                    balls = []
                    cx, cy, radius = None, None, None
                    if circles is not None:
                        circles = np.uint16(np.around(circles[0]))
                        for (x, y, r) in circles:
                            balls.append((int(x), int(y), int(r)))
                        lower_white = np.array([0, 0, 180], dtype=np.uint8)
                        upper_white = np.array([180, 50, 255], dtype=np.uint8)
                        mask_white = cv2.inRange(hsv, lower_white, upper_white)
                        mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask_green)
                        best_v = -1.0
                        for (x, y, r) in balls:
                            x0 = max(0, x - r // 2)
                            y0 = max(0, y - r // 2)
                            x1 = min(frame.shape[1], x + r // 2)
                            y1 = min(frame.shape[0], y + r // 2)
                            roi_hsv = hsv[y0:y1, x0:x1]
                            vmean = float(np.mean(roi_hsv[:, :, 2]))
                            smean = float(np.mean(roi_hsv[:, :, 1]))
                            if mask_white[y, x] > 0 and vmean > best_v and smean < 60:
                                best_v = vmean
                                cx, cy, radius = int(x), int(y), int(r)
                    if cx is None or cy is None:
                        time.sleep(0.1)
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rad = max(radius, 6)
                    roi_scale = int(rad * 4)
                    x0 = max(0, cx - roi_scale)
                    y0 = max(0, cy - roi_scale)
                    x1 = min(frame.shape[1] - 1, cx + roi_scale)
                    y1 = min(frame.shape[0] - 1, cy + roi_scale)
                    roi_gray = gray[y0:y1, x0:x1]
                    roi_gray = cv2.medianBlur(roi_gray, 3)
                    edges = cv2.Canny(roi_gray, 40, 110, apertureSize=3)
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=22, minLineLength=max(24, int(rad * 1.5)), maxLineGap=6)
                    vdir = None
                    stick_found = False
                    if lines is not None:
                        best_score = -1.0
                        best_vec = None
                        for ln in lines:
                            x1l, y1l, x2l, y2l = ln[0]
                            X1 = x0 + x1l
                            Y1 = y0 + y1l
                            X2 = x0 + x2l
                            Y2 = y0 + y2l
                            dx = X2 - X1
                            dy = Y2 - Y1
                            seg_len = float(np.hypot(dx, dy))
                            if seg_len < max(24, rad * 1.2):
                                continue
                            # distance from ball center to segment
                            vx = float(dx)
                            vy = float(dy)
                            if seg_len > 0:
                                ux = vx / seg_len
                                uy = vy / seg_len
                            else:
                                continue
                            wx = cx - X1
                            wy = cy - Y1
                            proj = wx * ux + wy * uy
                            proj = max(0.0, min(seg_len, proj))
                            closest_x = X1 + ux * proj
                            closest_y = Y1 + uy * proj
                            dist = float(np.hypot(closest_x - cx, closest_y - cy))
                            if dist > rad * 2.2:
                                continue
                            score = seg_len / (1.0 + dist)
                            if score > best_score:
                                best_score = score
                                # choose direction away from center
                                d1 = float(np.hypot(X1 - cx, Y1 - cy))
                                d2 = float(np.hypot(X2 - cx, Y2 - cy))
                                if d2 >= d1:
                                    best_vec = (dx, dy)
                                else:
                                    best_vec = (-dx, -dy)
                        if best_vec is not None:
                            bnorm = float(np.hypot(best_vec[0], best_vec[1]))
                            if bnorm > 0:
                                vdir = (best_vec[0] / bnorm, best_vec[1] / bnorm)
                                stick_found = True
                    if vdir is None:
                        vdir = self._estimate_direction_radial(gray, cx, cy, rad)
                    if circles is not None:
                        self.overlay._detected_result = (cx, cy, radius, vdir, balls, stick_found)
                    else:
                        self.overlay._detected_result = (cx, cy, radius, vdir, [], stick_found)
                except Exception:
                    time.sleep(0.1)
                time.sleep(0.033)

    def _estimate_direction_radial(self, gray, cx, cy, radius):
        h, w = gray.shape[:2]
        angles = np.deg2rad(np.arange(0, 360, 5))
        best_ang = 0.0
        best_val = 1e9
        for ang in angles:
            vals = []
            for t in range(radius + 5, radius + 80, 4):
                x = int(cx + np.cos(ang) * t)
                y = int(cy + np.sin(ang) * t)
                if x < 0 or y < 0 or x >= w or y >= h:
                    break
                vals.append(gray[y, x])
            if not vals:
                continue
            avg = float(np.mean(vals))
            if avg < best_val:
                best_val = avg
                best_ang = ang
        vx = np.cos(best_ang)
        vy = np.sin(best_ang)
        return (vx, vy)

class DraggableCircle(Widget):
    def __init__(self, radius=24, color_on=(0, 1, 0, 1), color_off=(1, 0, 0, 1), text_on="ON", text_off="OFF", on_release=None, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.color_on = color_on
        self.color_off = color_off
        self.text_on = text_on
        self.text_off = text_off
        self.on_release = on_release
        self.dragging = False
        self._is_on = True
        self.size = (radius * 2, radius * 2)
        self.label = Label(text=self.text_on, color=(0, 0, 0, 1), font_size=14, size_hint=(None, None))
        self.add_widget(self.label)
        self.bind(pos=self._redraw, size=self._redraw)
        self._redraw()

    def _redraw(self, *_):
        self.canvas.clear()
        c = self.color_on if self._is_on else self.color_off
        with self.canvas:
            Color(*c)
            Ellipse(pos=self.pos, size=self.size)
        self.label.text = self.text_on if self._is_on else self.text_off
        self.label.size = self.size
        self.label.pos = self.pos
        self.label.halign = "center"
        self.label.valign = "middle"
        self.label.texture_update()
        self.label.text_size = self.size

    def set_state(self, is_on):
        self._is_on = is_on
        self._redraw()

    @property
    def center(self):
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @center.setter
    def center(self, xy):
        cx, cy = xy
        self.pos = (cx - self.width / 2.0, cy - self.height / 2.0)

    def on_touch_down(self, touch):
        if self._hit(touch.x, touch.y):
            self.dragging = True
            return True
        return False

    def on_touch_move(self, touch):
        if self.dragging:
            nx = touch.x - self.width / 2.0
            ny = touch.y - self.height / 2.0
            nx = max(0, min(Window.width - self.width, nx))
            ny = max(0, min(Window.height - self.height, ny))
            self.pos = (nx, ny)
            return True
        return False

    def on_touch_up(self, touch):
        if self.dragging:
            self.dragging = False
            self._snap_to_nearest_corner()
            if self.on_release:
                self.on_release()
            return True
        return False

    def _hit(self, x, y):
        cx, cy = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 <= (self.radius) ** 2

    def _snap_to_nearest_corner(self):
        corners = [
            (self.width / 2.0, self.height / 2.0),
            (Window.width - self.width / 2.0, self.height / 2.0),
            (self.width / 2.0, Window.height - self.height / 2.0),
            (Window.width - self.width / 2.0, Window.height - self.height / 2.0),
        ]
        cx, cy = self.center
        dists = [((cx - px) ** 2 + (cy - py) ** 2) for (px, py) in corners]
        i = int(np.argmin(dists))
        self.center = corners[i]


class PoolHelperApp(App):
    def build(self):
        return OverlayWidget()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--image" and cv2 is not None:
        img_path = sys.argv[2]
        out_override = sys.argv[3] if len(sys.argv) >= 4 else None
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                frame = img.copy()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 40, 40], dtype=np.uint8)
                upper_green = np.array([85, 255, 255], dtype=np.uint8)
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                kernel = np.ones((5, 5), np.uint8)
                mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
                mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
                cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 10000:
                        hull = cv2.convexHull(largest)
                        poly_mask = np.zeros_like(mask_green)
                        cv2.fillConvexPoly(poly_mask, hull, 255)
                        mask_green = poly_mask
                gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray0 = cv2.medianBlur(gray0, 5)
                gray_masked = cv2.bitwise_and(gray0, gray0, mask=mask_green)
                mask_bgr = cv2.merge([mask_green, mask_green, mask_green])
                mask_out = os.path.join(os.path.dirname(img_path), "table_mask.png")
                cv2.imwrite(mask_out, mask_bgr)
                circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=15, minRadius=6, maxRadius=40)
                balls = []
                cx, cy, radius = None, None, None
                if circles is not None:
                    circles = np.uint16(np.around(circles[0]))
                    for (x, y, r) in circles:
                        balls.append((int(x), int(y), int(r)))
                    lower_white = np.array([0, 0, 180], dtype=np.uint8)
                    upper_white = np.array([180, 50, 255], dtype=np.uint8)
                    mask_white = cv2.inRange(hsv, lower_white, upper_white)
                    mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask_green)
                    best_v = -1.0
                    for (x, y, r) in balls:
                        x0 = max(0, x - r // 2)
                        y0 = max(0, y - r // 2)
                        x1 = min(frame.shape[1], x + r // 2)
                        y1 = min(frame.shape[0], y + r // 2)
                        roi_hsv = hsv[y0:y1, x0:x1]
                        vmean = float(np.mean(roi_hsv[:, :, 2]))
                        smean = float(np.mean(roi_hsv[:, :, 1]))
                        if mask_white[y, x] > 0 and vmean > best_v and smean < 60:
                            best_v = vmean
                            cx, cy, radius = int(x), int(y), int(r)
                if cx is None or cy is None:
                    h, w = frame.shape[:2]
                    sx = w // 2
                    sy = h // 2
                    vdir = (1.0, 0.0)
                    balls_list = []
                    cue_r = 12.0
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape[:2]
                    angles = np.deg2rad(np.arange(0, 360, 5))
                    best_ang = 0.0
                    best_val = 1e9
                    for ang in angles:
                        vals = []
                        for t in range(max(radius, 6) + 5, max(radius, 6) + 80, 4):
                            x = int(cx + np.cos(ang) * t)
                            y = int(cy + np.sin(ang) * t)
                            if x < 0 or y < 0 or x >= w or y >= h:
                                break
                            vals.append(gray[y, x])
                        if not vals:
                            continue
                        avg = float(np.mean(vals))
                        if avg < best_val:
                            best_val = avg
                            best_ang = ang
                    vx = np.cos(best_ang)
                    vy = np.sin(best_ang)
                    vdir = (float(vx), float(vy))
                    sx, sy = float(cx), float(cy)
                    balls_list = balls
                    cue_r = float(radius) if radius else 12.0
                def _draw_dashed(img, p1, p2, color, thickness=2, dash=12, gap=8):
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    dx, dy = x2 - x1, y2 - y1
                    seg_len = np.hypot(dx, dy)
                    if seg_len == 0:
                        return
                    ux, uy = dx / seg_len, dy / seg_len
                    dist = 0.0
                    while dist < seg_len:
                        d_end = min(dist + dash, seg_len)
                        sx1 = int(x1 + ux * dist)
                        sy1 = int(y1 + uy * dist)
                        sx2 = int(x1 + ux * d_end)
                        sy2 = int(y1 + uy * d_end)
                        cv2.line(img, (sx1, sy1), (sx2, sy2), color, thickness)
                        dist = d_end + gap
                def _image_paths_with_balls(img, sx, sy, vdir, balls, cue_r):
                    h, w = img.shape[:2]
                    cue_color = (255, 255, 255)
                    target_color = (255, 0, 0)
                    v = np.array(vdir, dtype=float)
                    m = np.linalg.norm(v)
                    if m == 0:
                        return
                    v /= m
                    pos = np.array([float(sx), float(sy)], dtype=float)
                    pockets = [
                        (0.0, 0.0, 20.0),
                        (w / 2.0, 0.0, 20.0),
                        (w, 0.0, 20.0),
                        (0.0, h, 20.0),
                        (w / 2.0, h, 20.0),
                        (w, h, 20.0),
                    ]
                    segments_drawn = 0
                    while segments_drawn < 6:
                        tx = np.inf if v[0] == 0 else ((w - pos[0]) / v[0] if v[0] > 0 else (0 - pos[0]) / v[0])
                        ty = np.inf if v[1] == 0 else ((h - pos[1]) / v[1] if v[1] > 0 else (0 - pos[1]) / v[1])
                        t_wall = min(tx, ty)
                        t_ball = np.inf
                        hit_ball = None
                        for (bx, by, br) in balls:
                            c = np.array([float(bx), float(by)], dtype=float)
                            r = float(br + cue_r)
                            oc = pos - c
                            b = 2.0 * np.dot(v, oc)
                            c2 = np.dot(oc, oc) - r * r
                            disc = b * b - 4.0 * c2
                            if disc <= 0:
                                continue
                            sqrt_disc = np.sqrt(disc)
                            t1 = (-b - sqrt_disc) / 2.0
                            t2 = (-b + sqrt_disc) / 2.0
                            t_hit = None
                            for tt in (t1, t2):
                                if tt > 1e-6:
                                    t_hit = tt if t_hit is None else min(t_hit, tt)
                            if t_hit is not None and t_hit < t_ball and t_hit < t_wall:
                                t_ball = t_hit
                                hit_ball = (bx, by, br)
                        t_pocket = np.inf
                        hit_pocket = None
                        for (px, py, pr) in pockets:
                            c = np.array([px, py], dtype=float)
                            r = float(pr)
                            oc = pos - c
                            b = 2.0 * np.dot(v, oc)
                            c2 = np.dot(oc, oc) - r * r
                            disc = b * b - 4.0 * c2
                            if disc <= 0:
                                continue
                            sqrt_disc = np.sqrt(disc)
                            t1 = (-b - sqrt_disc) / 2.0
                            t2 = (-b + sqrt_disc) / 2.0
                            t_hit = None
                            for tt in (t1, t2):
                                if tt > 1e-6:
                                    t_hit = tt if t_hit is None else min(t_hit, tt)
                            if t_hit is not None and t_hit < t_pocket and t_hit < t_ball and t_hit < t_wall:
                                t_pocket = t_hit
                                hit_pocket = (px, py, pr)
                        if hit_pocket is not None:
                            hit = pos + t_pocket * v
                            _draw_dashed(img, (pos[0], pos[1]), (hit[0], hit[1]), cue_color, 2)
                            cv2.circle(img, (int(hit_pocket[0]), int(hit_pocket[1])), int(hit_pocket[2]), (0, 255, 255), 2)
                            break
                        elif hit_ball is not None:
                            hit = pos + t_ball * v
                            _draw_dashed(img, (pos[0], pos[1]), (hit[0], hit[1]), cue_color, 2)
                            cv2.circle(img, (int(hit[0]), int(hit[1])), int(cue_r), (255, 255, 255), 2)
                            bx, by, br = hit_ball
                            n = np.array([float(bx), float(by)], dtype=float) - hit
                            nm = np.linalg.norm(n)
                            if nm == 0:
                                n = v.copy()
                                nm = 1.0
                            n /= nm
                            v_target = n
                            v_cue = v - n * np.dot(v, n)
                            _draw_ball_path(img, (bx, by), v_target, target_color, 6, w, h)
                            v = v_cue
                            pos = hit + v * 1e-7
                            segments_drawn += 1
                        else:
                            hit = pos + t_wall * v
                            _draw_dashed(img, (pos[0], pos[1]), (hit[0], hit[1]), cue_color, 2)
                            cv2.circle(img, (int(hit[0]), int(hit[1])), int(cue_r), (255, 255, 255), 2)
                            if tx < ty:
                                v[0] *= -1
                            else:
                                v[1] *= -1
                            pos = hit + v * 1e-7
                            segments_drawn += 1
                def _draw_ball_path(img, start_xy, vdir, color_bgr, max_segs, w, h):
                    pos = np.array([float(start_xy[0]), float(start_xy[1])], dtype=float)
                    v = np.array(vdir, dtype=float)
                    m = np.linalg.norm(v)
                    if m == 0:
                        return
                    v /= m
                    segs = 0
                    while segs < max_segs:
                        tx = np.inf if v[0] == 0 else ((w - pos[0]) / v[0] if v[0] > 0 else (0 - pos[0]) / v[0])
                        ty = np.inf if v[1] == 0 else ((h - pos[1]) / v[1] if v[1] > 0 else (0 - pos[1]) / v[1])
                        t = min(tx, ty)
                        if not np.isfinite(t) or t <= 0:
                            pos = pos + v * 1e-6
                            continue
                        hit = pos + t * v
                        _draw_dashed(img, (pos[0], pos[1]), (hit[0], hit[1]), color_bgr, 2)
                        eps = 1e-7
                        if tx < ty:
                            v[0] *= -1
                        else:
                            v[1] *= -1
                        pos = hit + v * eps
                        segs += 1
                # Draw red circles over all detected balls for debug
                for (bx, by, br) in balls_list:
                    cv2.circle(img, (int(bx), int(by)), int(br), (0, 0, 255), 2)
                _image_paths_with_balls(img, sx, sy, vdir, balls_list, cue_r)
                if out_override is not None:
                    out_path = os.path.join(os.path.dirname(img_path), out_override)
                else:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    if base.startswith("image_"):
                        out_path = os.path.join(os.path.dirname(img_path), "final_debug.png")
                    else:
                        out_path = os.path.join(os.path.dirname(img_path), base + "_annotated.png")
                cv2.imwrite(out_path, img)
                print(out_path)
        else:
            print("Image not found")
    else:
        PoolHelperApp().run()
