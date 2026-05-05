import React, { useCallback, useEffect, useRef, useState } from "react";

import {
  MoonIcon,
  ShieldCheckIcon,
  ShieldExclamationIcon,
  SunIcon,
} from "@heroicons/react/24/outline";

import { fetchJson } from "./fetchJson";
import { NavigatorOptionCard } from "./NavigatorOptionCard";
import { PolarisIconSwitch } from "./PolarisIconSwitch";

type TwistCommand = {
  linear_x?: number;
  linear_y?: number;
  angular_z?: number;
};

type TeleopButtonId = "w" | "a" | "s" | "d" | "q" | "e" | "stop";

type ControlMode = "passive" | "manual" | "autonomous";

const SPEED_LINEAR = 0.4; // m/s
const SPEED_ANGULAR = 0.6; // rad/s
/**
 * The Go2 WebRTC connection (`UnitreeWebRTCConnection.cmd_vel_timeout = 0.2`) drops
 * stale commands fast — we re-send on a tight interval while the button is held.
 */
const REPEAT_MS = 80;

const COMMAND_BY_KEY: Record<TeleopButtonId, TwistCommand> = {
  w: { linear_x: SPEED_LINEAR },
  s: { linear_x: -SPEED_LINEAR },
  q: { linear_y: SPEED_LINEAR },
  e: { linear_y: -SPEED_LINEAR },
  a: { angular_z: SPEED_ANGULAR },
  d: { angular_z: -SPEED_ANGULAR },
  stop: {},
};

const KEY_TO_BUTTON: Record<string, TeleopButtonId | undefined> = {
  w: "w",
  a: "a",
  s: "s",
  d: "d",
  q: "q",
  e: "e",
  " ": "stop",
};

async function sendTeleop(
  command: TwistCommand,
  signal: AbortSignal,
): Promise<void> {
  try {
    await fetchJson("/api/teleop/command", {
      method: "POST",
      body: JSON.stringify({
        linear_x: command.linear_x ?? 0,
        linear_y: command.linear_y ?? 0,
        linear_z: 0,
        angular_x: 0,
        angular_y: 0,
        angular_z: command.angular_z ?? 0,
      }),
      signal,
    });
  } catch (error) {
    if ((error as { name?: string }).name === "AbortError") {
      return;
    }
  }
}

async function sendStop(signal?: AbortSignal): Promise<void> {
  try {
    await fetchJson("/api/teleop/stop", { method: "POST", signal });
  } catch (error) {
    if ((error as { name?: string }).name === "AbortError") {
      return;
    }
  }
}

async function setSearchlight(enabled: boolean): Promise<string> {
  const result = await fetchJson<{ ok: boolean; result?: string }>(
    "/api/searchlight",
    {
      method: "POST",
      body: JSON.stringify({ enabled }),
    },
  );
  return result.result ?? (enabled ? "Searchlight on" : "Searchlight off");
}

async function setObstacleAvoidance(enabled: boolean): Promise<string> {
  const result = await fetchJson<{ ok: boolean; result?: string }>(
    "/api/obstacle-avoidance",
    {
      method: "POST",
      body: JSON.stringify({ enabled }),
    },
  );
  return (
    result.result ??
    (enabled ? "Obstacle avoidance on" : "Obstacle avoidance off")
  );
}

async function startExploration(): Promise<string> {
  const result = await fetchJson<{ ok: boolean; result?: string }>(
    "/api/explore/start",
    { method: "POST" },
  );
  return result.result ?? "Started";
}

async function stopExploration(): Promise<string> {
  const result = await fetchJson<{ ok: boolean; result?: string }>(
    "/api/explore/stop",
    { method: "POST" },
  );
  return result.result ?? "Stopped";
}

export type NavigatorTeleopPanelProps = {
  connected: boolean;
};

const MODE_LABEL: Record<ControlMode, string> = {
  passive: "Passive",
  manual: "Manual",
  autonomous: "Autonomous",
};

const MODE_HINT: Record<ControlMode, string> = {
  passive:
    "Robot stays put. LiDAR maps whatever's within sensor range from this position.",
  manual: "Hold a key or button to walk · release to stop",
  autonomous:
    "Robot drives itself toward the nearest unmapped frontier and keeps going until stopped.",
};

/**
 * Compact control surface in the navigator sidebar — three mapping/control modes:
 *
 * - **Passive**: no motion command issued; LiDAR maps from a stationary pose.
 * - **Manual**: WASD pad. Hold to walk; release to stop. Re-publishes every 80 ms so
 *   the Go2 WebRTC bridge keeps moving past its 200 ms `cmd_vel_timeout`.
 * - **Autonomous**: invokes `WavefrontFrontierExplorer` via MCP (`begin_exploration`
 *   / `end_exploration`) so the robot drives itself to unmapped frontiers.
 *
 * Switching modes always issues a stop first so motion never carries between modes.
 */
export function NavigatorTeleopPanel(
  props: NavigatorTeleopPanelProps,
): React.ReactElement {
  const { connected } = props;

  const [mode, setMode] = useState<ControlMode>("manual");
  const [exploreBusy, setExploreBusy] = useState(false);
  const [exploreActive, setExploreActive] = useState(false);
  const [exploreStatus, setExploreStatus] = useState<string | null>(null);
  const [searchlightOn, setSearchlightOn] = useState(false);
  const [searchlightBusy, setSearchlightBusy] = useState(false);
  // Obstacle avoidance defaults ON — the Go2's onboard avoidance is the safety
  // net while teleoperating; flipping off should be deliberate.
  const [avoidOn, setAvoidOn] = useState(true);
  const [avoidBusy, setAvoidBusy] = useState(false);

  const activeRef = useRef<TeleopButtonId | null>(null);
  const intervalRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [activeButton, setActiveButton] = useState<TeleopButtonId | null>(
    null,
  );

  const stopAllManual = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (abortRef.current) {
      abortRef.current.abort();
    }
    abortRef.current = new AbortController();
    void sendStop(abortRef.current.signal);
    activeRef.current = null;
    setActiveButton(null);
  }, []);

  const startCommand = useCallback(
    (id: TeleopButtonId) => {
      if (!connected || mode !== "manual") {
        return;
      }
      if (id === "stop") {
        stopAllManual();
        return;
      }
      if (activeRef.current === id) {
        return;
      }
      activeRef.current = id;
      setActiveButton(id);
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (abortRef.current) {
        abortRef.current.abort();
      }
      abortRef.current = new AbortController();
      const command = COMMAND_BY_KEY[id];
      const signal = abortRef.current.signal;
      void sendTeleop(command, signal);
      intervalRef.current = window.setInterval(() => {
        void sendTeleop(command, signal);
      }, REPEAT_MS);
    },
    [connected, mode, stopAllManual],
  );

  const releaseCommand = useCallback(
    (id: TeleopButtonId) => {
      if (id === "stop") {
        return;
      }
      if (activeRef.current !== id) {
        return;
      }
      stopAllManual();
    },
    [stopAllManual],
  );

  // Stop everything when leaving manual mode or disconnecting.
  useEffect(() => {
    if (mode !== "manual" || !connected) {
      stopAllManual();
    }
  }, [mode, connected, stopAllManual]);

  // Stop autonomous exploration when leaving that mode or disconnecting.
  useEffect(() => {
    if (mode === "autonomous" && connected) {
      return;
    }
    if (exploreActive) {
      setExploreActive(false);
      void stopExploration().catch(() => undefined);
    }
  }, [mode, connected, exploreActive]);

  // Keyboard shortcuts — manual mode only.
  useEffect(() => {
    if (!connected || mode !== "manual") {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      const id = KEY_TO_BUTTON[event.key.toLowerCase()];
      if (!id) {
        return;
      }
      event.preventDefault();
      startCommand(id);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const id = KEY_TO_BUTTON[event.key.toLowerCase()];
      if (!id || id === "stop") {
        return;
      }
      releaseCommand(id);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [connected, mode, startCommand, releaseCommand]);

  // On unmount — make absolutely sure everything stops.
  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }
      if (abortRef.current) {
        abortRef.current.abort();
      }
      void sendStop();
      void stopExploration().catch(() => undefined);
    };
  }, []);

  const handleModeChange = useCallback(
    (next: ControlMode) => {
      // Always stop motion crossing modes — never carry teleop or exploration over.
      if (mode === "manual") {
        stopAllManual();
      }
      if (mode === "autonomous" && exploreActive) {
        setExploreActive(false);
        void stopExploration().catch(() => undefined);
      }
      setMode(next);
      setExploreStatus(null);
    },
    [mode, exploreActive, stopAllManual],
  );

  const handleStartExplore = useCallback(async () => {
    if (!connected || exploreBusy) {
      return;
    }
    setExploreBusy(true);
    try {
      const text = await startExploration();
      setExploreActive(true);
      setExploreStatus(text);
    } catch (error) {
      setExploreStatus((error as Error).message ?? "Failed to start exploration");
    } finally {
      setExploreBusy(false);
    }
  }, [connected, exploreBusy]);

  const handleToggleSearchlight = useCallback(async () => {
    if (!connected || searchlightBusy) {
      return;
    }
    const next = !searchlightOn;
    setSearchlightBusy(true);
    // Optimistic flip — revert if request fails so the toggle reflects truth.
    setSearchlightOn(next);
    try {
      await setSearchlight(next);
    } catch {
      setSearchlightOn(!next);
    } finally {
      setSearchlightBusy(false);
    }
  }, [connected, searchlightBusy, searchlightOn]);

  // When the robot disconnects, reflect that the light is off (firmware won't keep it lit).
  useEffect(() => {
    if (!connected && searchlightOn) {
      setSearchlightOn(false);
    }
  }, [connected, searchlightOn]);

  const handleToggleAvoid = useCallback(async () => {
    if (!connected || avoidBusy) {
      return;
    }
    const next = !avoidOn;
    setAvoidBusy(true);
    setAvoidOn(next);
    try {
      await setObstacleAvoidance(next);
    } catch {
      setAvoidOn(!next);
    } finally {
      setAvoidBusy(false);
    }
  }, [connected, avoidBusy, avoidOn]);

  const handleStopExplore = useCallback(async () => {
    if (exploreBusy) {
      return;
    }
    setExploreBusy(true);
    try {
      const text = await stopExploration();
      setExploreActive(false);
      setExploreStatus(text);
    } catch (error) {
      setExploreStatus((error as Error).message ?? "Failed to stop exploration");
    } finally {
      setExploreBusy(false);
    }
  }, [exploreBusy]);

  function teleopBtnProps(id: TeleopButtonId) {
    return {
      "aria-label": teleopAriaLabel(id),
      "aria-pressed": activeButton === id,
      "data-teleop": id,
      className: [
        "polaris-nav-teleop-btn",
        `polaris-nav-teleop-btn--${id}`,
        activeButton === id ? "is-active" : "",
      ]
        .filter(Boolean)
        .join(" "),
      disabled: !connected || mode !== "manual",
      type: "button" as const,
      onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        startCommand(id);
      },
      onPointerUp: (event: React.PointerEvent<HTMLButtonElement>) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        releaseCommand(id);
      },
      onPointerLeave: () => {
        releaseCommand(id);
      },
      onPointerCancel: () => {
        releaseCommand(id);
      },
    };
  }

  return (
    <NavigatorOptionCard
      className="polaris-nav-teleop-card polaris-nav-option-card--polaris-heading"
      headerAside={
        <select
          aria-label="Control mode"
          className="polaris-nav-teleop-mode-select"
          disabled={!connected}
          onChange={(event) => {
            handleModeChange(event.target.value as ControlMode);
          }}
          value={mode}
        >
          <option value="passive">{MODE_LABEL.passive}</option>
          <option value="manual">{MODE_LABEL.manual}</option>
          <option value="autonomous">{MODE_LABEL.autonomous}</option>
        </select>
      }
      title="Control"
    >
      <div className="polaris-nav-teleop-shell">
        <p className="polaris-nav-teleop-hint">
          {connected
            ? MODE_HINT[mode]
            : "Connect a robot to enable control"}
        </p>

        {mode === "manual" ? (
          <>
            <div
              aria-label="Teleop pad"
              className="polaris-nav-teleop-grid"
              role="group"
            >
              <button {...teleopBtnProps("q")}>
                <span className="polaris-nav-teleop-glyph">Q</span>
                <span className="polaris-nav-teleop-sub">strafe ←</span>
              </button>
              <button {...teleopBtnProps("w")}>
                <span className="polaris-nav-teleop-glyph">W</span>
                <span className="polaris-nav-teleop-sub">forward</span>
              </button>
              <button {...teleopBtnProps("e")}>
                <span className="polaris-nav-teleop-glyph">E</span>
                <span className="polaris-nav-teleop-sub">strafe →</span>
              </button>
              <button {...teleopBtnProps("a")}>
                <span className="polaris-nav-teleop-glyph">A</span>
                <span className="polaris-nav-teleop-sub">turn ←</span>
              </button>
              <button {...teleopBtnProps("s")}>
                <span className="polaris-nav-teleop-glyph">S</span>
                <span className="polaris-nav-teleop-sub">back</span>
              </button>
              <button {...teleopBtnProps("d")}>
                <span className="polaris-nav-teleop-glyph">D</span>
                <span className="polaris-nav-teleop-sub">turn →</span>
              </button>
            </div>
            <button
              {...teleopBtnProps("stop")}
              className="polaris-nav-teleop-stop"
            >
              Stop (Space)
            </button>
          </>
        ) : null}

        {mode === "autonomous" ? (
          <div className="polaris-nav-teleop-explore">
            {!exploreActive ? (
              <button
                className="polaris-nav-teleop-explore-start"
                disabled={!connected || exploreBusy}
                onClick={() => {
                  void handleStartExplore();
                }}
                type="button"
              >
                {exploreBusy ? "Starting…" : "Start exploration"}
              </button>
            ) : (
              <button
                className="polaris-nav-teleop-explore-stop"
                disabled={exploreBusy}
                onClick={() => {
                  void handleStopExplore();
                }}
                type="button"
              >
                {exploreBusy ? "Stopping…" : "Stop exploration"}
              </button>
            )}
            {exploreStatus ? (
              <p className="polaris-nav-teleop-explore-status">
                {exploreStatus}
              </p>
            ) : null}
          </div>
        ) : null}

        {mode === "passive" ? (
          <div className="polaris-nav-teleop-passive">
            <span className="polaris-nav-teleop-passive-dot" aria-hidden />
            <span>Idle · waiting on LiDAR sweeps</span>
          </div>
        ) : null}

        <div
          aria-label="Robot toggles"
          className="polaris-nav-teleop-toggles"
          role="group"
        >
          <div
            className={[
              "polaris-nav-teleop-toggle",
              "polaris-nav-teleop-toggle--searchlight",
              "polaris-nav-teleop-toggle--switch",
              searchlightOn ? "is-on" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <span className="polaris-nav-teleop-toggle-label">Searchlight</span>
            <PolarisIconSwitch
              ariaLabel={
                searchlightOn ? "Turn searchlight off" : "Turn searchlight on"
              }
              checked={searchlightOn}
              disabled={!connected || searchlightBusy}
              offIcon={<SunIcon aria-hidden className="polaris-icon-switch-icon" />}
              onCheckedChange={(next) => {
                if (next === searchlightOn) {
                  return;
                }
                void handleToggleSearchlight();
              }}
              onIcon={<MoonIcon aria-hidden className="polaris-icon-switch-icon" />}
            />
          </div>

          <div
            className={[
              "polaris-nav-teleop-toggle",
              "polaris-nav-teleop-toggle--avoid",
              "polaris-nav-teleop-toggle--switch",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <span className="polaris-nav-teleop-toggle-label">
              Obstacle avoid
            </span>
            <PolarisIconSwitch
              ariaLabel={
                avoidOn
                  ? "Disable obstacle avoidance"
                  : "Enable obstacle avoidance"
              }
              checked={avoidOn}
              disabled={!connected || avoidBusy}
              offIcon={
                <ShieldExclamationIcon
                  aria-hidden
                  className="polaris-icon-switch-icon"
                />
              }
              onCheckedChange={(next) => {
                if (next === avoidOn) {
                  return;
                }
                void handleToggleAvoid();
              }}
              onIcon={
                <ShieldCheckIcon
                  aria-hidden
                  className="polaris-icon-switch-icon"
                />
              }
            />
          </div>
        </div>
      </div>
    </NavigatorOptionCard>
  );
}

function teleopAriaLabel(id: TeleopButtonId): string {
  switch (id) {
    case "w":
      return "Walk forward";
    case "s":
      return "Walk backward";
    case "a":
      return "Turn left";
    case "d":
      return "Turn right";
    case "q":
      return "Strafe left";
    case "e":
      return "Strafe right";
    case "stop":
      return "Stop";
  }
}
