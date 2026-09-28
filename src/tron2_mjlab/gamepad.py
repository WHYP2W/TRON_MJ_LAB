"""Xbox input through pygame's SDL controller mapping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def normalize_axis(value: int, deadzone: float) -> float:
    if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be finite and in [0, 1)")
    normalized = value / (32768.0 if value < 0 else 32767.0)
    normalized = max(-1.0, min(1.0, normalized))
    if abs(normalized) <= deadzone:
        return 0.0
    return math.copysign(
        (abs(normalized) - deadzone) / (1.0 - deadzone), normalized
    )


@dataclass(frozen=True)
class GamepadInput:
    connected: bool = False
    forward: float = 0.0
    turn: float = 0.0
    arm_horizontal: float = 0.0
    arm_vertical: float = 0.0
    gripper: float = 0.0
    triggers_released: bool = True
    enable: bool = False
    stop: bool = False
    previous_pair: bool = False
    next_pair: bool = False
    home: bool = False
    reset: bool = False
    pause: bool = False


@dataclass(frozen=True)
class GamepadCommand:
    enabled: bool = False
    forward: float = 0.0
    turn: float = 0.0
    arm_horizontal: float = 0.0
    arm_vertical: float = 0.0
    gripper: float = 0.0
    arm_pair: int = 0
    home: bool = False
    reset: bool = False
    pause: bool = False


class GamepadControl:
    def __init__(self) -> None:
        self.arm_pair = 0
        self.command = GamepadCommand()
        self._previous = GamepadInput()
        self._ready = False

    def disarm(self) -> None:
        self._ready = False
        self.command = GamepadCommand(arm_pair=self.arm_pair)

    def update(self, state: GamepadInput) -> GamepadCommand:
        previous = self._previous
        self._previous = state
        if not state.connected:
            self.disarm()
            return self.command
        if not previous.connected:
            self.disarm()

        reset = state.reset and not previous.reset
        pause = state.pause and not previous.pause
        axes = (
            state.forward,
            state.turn,
            state.arm_horizontal,
            state.arm_vertical,
            state.gripper,
        )
        if (
            not state.enable
            and not any(axes)
            and state.triggers_released
        ):
            self._ready = True
        if state.stop or reset or pause:
            self.disarm()

        if state.previous_pair and not previous.previous_pair:
            self.arm_pair = (self.arm_pair - 1) % 3
        if state.next_pair and not previous.next_pair:
            self.arm_pair = (self.arm_pair + 1) % 3

        enabled = self._ready and state.enable
        self.command = GamepadCommand(
            enabled=enabled,
            forward=state.forward if enabled else 0.0,
            turn=state.turn if enabled else 0.0,
            arm_horizontal=state.arm_horizontal if enabled else 0.0,
            arm_vertical=state.arm_vertical if enabled else 0.0,
            gripper=state.gripper if enabled else 0.0,
            arm_pair=self.arm_pair,
            home=enabled and state.home and not previous.home,
            reset=reset,
            pause=pause,
        )
        return self.command


class PygameGamepad:
    def __init__(
        self, controller_index: int | None = None, deadzone: float = 0.15
    ) -> None:
        normalize_axis(0, deadzone)
        if controller_index is not None and controller_index < 0:
            raise ValueError("controller_index must be nonnegative")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

        import pygame
        from pygame._sdl2 import controller

        self._pygame = pygame
        self._sdl = controller
        self._controller: controller.Controller | None = None
        self._index = controller_index
        self._deadzone = deadzone
        self._scan = True
        self.name: str | None = None
        try:
            pygame.display.init()
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            controller.init()
        except Exception:
            self.close()
            raise

    def devices(self) -> list[tuple[int, str]]:
        self._pygame.event.pump()
        return [
            (index, self._sdl.name_forindex(index) or f"Controller {index}")
            for index in range(self._sdl.get_count())
            if self._sdl.is_controller(index)
        ]

    def _disconnect(self) -> None:
        if self._controller is not None:
            self._controller.quit()
            self._controller = None
        self.name = None

    def poll(self) -> GamepadInput:
        pygame = self._pygame
        try:
            events = pygame.event.get()
            if any(
                event.type == pygame.CONTROLLERDEVICEADDED for event in events
            ):
                self._scan = True
            if (
                self._controller is not None
                and not self._controller.attached()
            ):
                self._disconnect()
                self._scan = True
                print("[GAMEPAD] Disconnected. Motion commands stopped.")
                return GamepadInput()
            if self._controller is None and self._scan:
                self._scan = False
                for index, name in self.devices():
                    if self._index is None or index == self._index:
                        self._controller = self._sdl.Controller(index)
                        self.name = name
                        print(f"[GAMEPAD] Connected: {name} (index {index})")
                        break
            if self._controller is None:
                return GamepadInput()

            def axis(axis_id: int) -> float:
                assert self._controller is not None
                return normalize_axis(
                    self._controller.get_axis(axis_id), self._deadzone
                )

            button = self._controller.get_button
            left_trigger = axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
            right_trigger = axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
            return GamepadInput(
                connected=True,
                forward=-axis(pygame.CONTROLLER_AXIS_LEFTY),
                turn=-axis(pygame.CONTROLLER_AXIS_LEFTX),
                arm_horizontal=axis(pygame.CONTROLLER_AXIS_RIGHTX),
                arm_vertical=-axis(pygame.CONTROLLER_AXIS_RIGHTY),
                gripper=right_trigger - left_trigger,
                triggers_released=(
                    left_trigger == 0.0 and right_trigger == 0.0
                ),
                enable=bool(button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER)),
                stop=bool(button(pygame.CONTROLLER_BUTTON_B)),
                previous_pair=bool(button(pygame.CONTROLLER_BUTTON_DPAD_LEFT)),
                next_pair=bool(button(pygame.CONTROLLER_BUTTON_DPAD_RIGHT)),
                home=bool(button(pygame.CONTROLLER_BUTTON_A)),
                reset=bool(button(pygame.CONTROLLER_BUTTON_BACK)),
                pause=bool(button(pygame.CONTROLLER_BUTTON_START)),
            )
        except pygame.error as error:
            self._disconnect()
            print(
                f"[GAMEPAD] Input unavailable: {error}. "
                "Motion commands stopped."
            )
            return GamepadInput()

    def close(self) -> None:
        self._disconnect()
        self._sdl.quit()
        self._pygame.display.quit()
