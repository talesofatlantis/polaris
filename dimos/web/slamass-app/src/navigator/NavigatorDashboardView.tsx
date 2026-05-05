import { CameraIcon } from "@heroicons/react/24/outline";
import React, {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { AgentToolsModal } from "../AgentToolsModal";
import { LiveFeedPanel, type LiveFeedPanelHandle } from "../LiveFeedPanel";
import { MapPane, type MapAdditionalRobotMarker } from "../MapPane";
import { NAVIGATOR_MAP_VIEWPORT_ANCHOR_Y } from "../mapViewport";
import {
  buildFleetEntryFromDeploy,
  fleetEntriesToHydrateJson,
  parseHydratedFleetEntries,
} from "../polarisDeployFleetEntry";
import {
  readPolarisDeployHydratedFleetJson,
  takePolarisDeployPendingPayload,
  writePolarisDeployHydratedFleetJson,
} from "../polarisDeploySession";
import {
  POLARIS_GO2_PREVIEW_URL,
  POLARIS_OPERATOR_SELECT_THUMB_URL,
} from "../polarisAssets";
import type { PolarisOperatorFleetEntry } from "../polarisOperatorFleet";
import {
  defaultRobotOperatorHoverCard,
  robotOperatorHoverCardFromPolarisFleetEntry,
} from "../robotOperatorLabel";
import { OperatorRail, SelectedSemanticPreview } from "../OperatorRail";
import { PanelShell } from "../PanelShell";
import { SettingsCogGlyphs } from "../SettingsCogGlyphs";
import {
  buildSemanticItems,
  isPolarisDemoCameraCaptureRef,
  polarisDemoCameraCapturePreviewFields,
  resolveSelectedPoi,
  resolveSelectedYoloObject,
} from "../semanticItems";
import {
  calculateTeleopCommand,
  isEditableTarget,
  normalizeTeleopKey,
  PUBLISH_RATE_HZ,
  teleopKeys,
} from "../teleop";
import type { ChatToolDefinition, ManualInspectionMode } from "../types";
import { fetchJson } from "./fetchJson";
import { NavigatorMapControlsHover } from "./NavigatorMapControlsHover";
import { NavigatorOperatorFleet } from "./NavigatorOperatorFleet";
import { NavigatorOptionCard } from "./NavigatorOptionCard";
import { NavigatorTeleopPanel } from "./NavigatorTeleopPanel";
import type { ActivityEntry } from "./useNavigatorSlamassState";
import { useNavigatorSlamassState } from "./useNavigatorSlamassState";

type NavigatorDashboardViewProps = ReturnType<
  typeof useNavigatorSlamassState
>;

function formatYaw(yaw: number): string {
  return `${Math.round((yaw * 180) / Math.PI)}°`;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return "No data";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatNavigatorActivityLine(entry: ActivityEntry): string {
  const detail = entry.detail.trim();
  if (detail.length === 0) {
    return `${entry.timestamp} · ${entry.title}`;
  }
  return `${entry.timestamp} · ${entry.title} — ${detail}`;
}

function initialDeployedOperators(): PolarisOperatorFleetEntry[] {
  if (typeof window === "undefined") {
    return [];
  }
  const raw = readPolarisDeployHydratedFleetJson();
  if (!raw) {
    return [];
  }
  return parseHydratedFleetEntries(raw);
}

function navigatorActivityToneClass(tone: ActivityEntry["tone"]): string {
  switch (tone) {
    case "danger":
      return "tone-danger";
    case "accent":
      return "tone-accent";
    case "success":
      return "tone-success";
    default:
      return "polaris-navigator-activity-line--neutral";
  }
}

export function NavigatorDashboardView(
  props: NavigatorDashboardViewProps,
): React.ReactElement {
  const {
    state,
    slamassApiStatus,
    busyAction,
    activityEntries,
    queueCameraSync,
    handleSelectItem,
    handleFocusItem,
    handleFocusMap,
    handleFocusRobot,
    handleClearFocus,
    handleNavigate,
    handleHighlightItem,
    handleGoToItem,
    handleDeleteItem,
    handleSubmitChatMessage,
    handleResetChat,
    handleInspectNow,
    handleSaveMap,
    handleInspectionModeChange,
    handleYoloModeChange,
    handleYoloInferenceEnabledChange,
    handleLayerToggle,
    handleClearLowLevelMapMemory,
    handleClearSemanticMemory,
    handleStopDimos,
    handleStopMotion,
    appendActivity,
    reportActionError,
  } = props;

  const [go2OperatorFleetHover, setGo2OperatorFleetHover] = useState(false);
  const [deployedOperators, setDeployedOperators] = useState<PolarisOperatorFleetEntry[]>(() =>
    initialDeployedOperators(),
  );
  const [robotMarkerDeployPop, setRobotMarkerDeployPop] = useState(false);
  const [controlsMenuOpen, setControlsMenuOpen] = useState(false);
  const [teleopEnabled, setTeleopEnabled] = useState(false);
  const [agentToolsOpen, setAgentToolsOpen] = useState(false);
  const [agentToolsLoading, setAgentToolsLoading] = useState(false);
  const [agentToolsError, setAgentToolsError] = useState<string | null>(null);
  const [agentTools, setAgentTools] = useState<ChatToolDefinition[] | null>(
    null,
  );

  const latestActivity =
    activityEntries.length === 0
      ? null
      : activityEntries[activityEntries.length - 1];

  const appendActivityRef = useRef(appendActivity);
  appendActivityRef.current = appendActivity;

  useEffect(() => {
    const pending = takePolarisDeployPendingPayload();
    if (!pending) {
      return;
    }
    const entry = buildFleetEntryFromDeploy(pending);
    if (!entry) {
      return;
    }
    writePolarisDeployHydratedFleetJson(fleetEntriesToHydrateJson([entry]));
    setDeployedOperators([entry]);
    setRobotMarkerDeployPop(true);
    appendActivityRef.current(
      "system",
      "Operator deployed",
      `${entry.title} is live on the map.`,
      "success",
    );
    const endPop = window.setTimeout(() => {
      setRobotMarkerDeployPop(false);
    }, 2800);
    return () => {
      window.clearTimeout(endPop);
    };
  }, []);

  const controlsMenuRef = useRef<HTMLDivElement>(null);
  const cameraFeedRef = useRef<LiveFeedPanelHandle>(null);
  const [cameraHeaderCaptureEnabled, setCameraHeaderCaptureEnabled] =
    useState(false);
  const teleopIntervalRef = useRef<number | null>(null);
  const teleopKeysRef = useRef<Set<string>>(new Set());
  const teleopRequestInFlightRef = useRef(false);
  const teleopErrorMessageRef = useRef<string | null>(null);

  const selectedPoi = React.useMemo(
    () => resolveSelectedPoi(state.pois, state.ui.selected_item),
    [state.pois, state.ui.selected_item],
  );
  const selectedYoloObject = React.useMemo(
    () => resolveSelectedYoloObject(state.yolo_objects, state.ui.selected_item),
    [state.yolo_objects, state.ui.selected_item],
  );
  const semanticItems = React.useMemo(
    () => buildSemanticItems(state.pois, state.yolo_objects),
    [state.pois, state.yolo_objects],
  );

  const selectedPreview = React.useMemo<SelectedSemanticPreview | null>(() => {
    if (state.ui.selected_item && isPolarisDemoCameraCaptureRef(state.ui.selected_item)) {
      return polarisDemoCameraCapturePreviewFields();
    }
    if (selectedPoi) {
      return {
        kind: "vlm_poi",
        entity_id: selectedPoi.poi_id,
        title: selectedPoi.title,
        subtitle: selectedPoi.category,
        summary: selectedPoi.summary,
        thumbnail_url: selectedPoi.thumbnail_url,
      };
    }
    if (selectedYoloObject) {
      return {
        kind: "yolo_object",
        entity_id: selectedYoloObject.object_id,
        title: selectedYoloObject.label,
        subtitle: `${Math.round(selectedYoloObject.best_confidence * 100)}% confidence`,
        summary: `${selectedYoloObject.detections_count} linked detections. Best view stored for revisit.`,
        thumbnail_url: selectedYoloObject.thumbnail_url,
      };
    }
    return null;
  }, [selectedPoi, selectedYoloObject, state.ui.selected_item]);

  /**
   * Fallback mock costmap — matches the visual style of the saved sim costmap
   * (irregular slate-100 blob with darker observed patches on a white field).
   * Anchored to a fixed world origin so `MapPane`'s structure key stays stable
   * across robot-pose updates (otherwise the fade-in keeps resetting and the
   * canvas stays at opacity 0). Real costmap deltas from slamass replace this
   * as soon as they arrive.
   *
   * The 118 × 113 grid + 0.15 m resolution + (-8.85, -5.25) origin mirror the
   * dimensions of the sim's saved map so the bounded shape lands naturally
   * inside the navigator's viewport at default zoom.
   */
  const FALLBACK_MAP = useMemo(
    () => ({
      map_id: "fallback",
      // Real saved costmap PNG captured from the dimos `--replay` session
      // (122 × 127 cells @ 0.075 m). Embedded below as base64 so the navigator
      // panel always shows the same visual style as the original sim view,
      // regardless of whether the live perception pipeline has produced a
      // fresh costmap yet.
      //
      // The recorded origin was (-2.775, 0.3); we re-anchor the placeholder
      // so the world origin (0, 0) sits in the middle of the map — a real
      // Go2 typically boots near (0, 0), so this keeps it visible and lets
      // the live pose marker walk a sensible distance before reaching the
      // placeholder's edge. As the robot moves, its `state.robot_pose`
      // updates the marker on this fixed canvas.
      resolution: 0.075,
      origin_x: -((122 * 0.075) / 2),
      origin_y: -((127 * 0.075) / 2),
      width: 122,
      height: 127,
      updated_at: new Date(0).toISOString(),
      image_version: 0,
      image_url:
        "data:image/png;base64,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1uVPT09Nz1/T+nq0fn2QbOmO2Vz0lBU0AAQS0AAQSEADQKBd/35/C8r7QaftZV1qPX/vueyyg3748aPr87cqO+jzcz9ySdbeJyBd674Ba89Bj9ZZW0EDQCABDQCBBDQABDIHvTG1jqz3Xtzl43V279M5b4v37zxrd86jf/+soAEgkIAGgEACGgAC6aA3Zm7n07szWruDKq3d+Zbn733/afqq7Q2/9vtpa2rv5+PxODl+fXmZdb60f396s4IGgEACGgACCWgACKRg2Zh//PN+stn0xeXl5OO9555LvffebX2+3l9Pb2WHWdsL/OrqanL88fq6+zXx+1rff2WHWv48Ms+nw+2uM8oKGgACCWgACCSgASDQrn+/P4LyftFz9e6ge98/uvX5e59vLnPPY9FBn9beOmkraAAIJKABIJCABoBAu/p9/h7M7aR7d7St96feWiddzjE/PT01PV4nTZK9dbzprKABIJCABoBAAhoAAukbGrV2vKfudE49F106dQfcey776fFxcvxYHC99fliTDjqLFTQABBLQABBIQANAoA9rX8DWlB1N2fnurcNZuoP9+uXzu69n78691jmX1t7bGxiXFTQABBLQABBIQANAoF31pdSdeo76+vp6cvyh2Nu61vn3Vrvei4vp/2lvD4emx5fMRbOmvf3NzNZYQQNAIAENAIEENAAEMgfNqsrOuebUnXTp9fV1cly7v/XDjx+T47e3k14uTOict8UKGgACCWgACCSgASCQDjpc7451tA6qdyfdurd4yd7cQC9W0AAQSEADQCABDQCBdNAVOuD3lV/Pt+9/TV6vWqdbmyNeW+167L0NLMUKGgACCWgACCSgASCQDrqzvc3Blp1zqfZ66GQB/p4VNAAEEtAAEEhAA0AgHfRMrXPA5ce/fvm86v2N52rdu7rWOW9tTrz29evYgT9lBQ0AgQQ0AAQS0AAQaFN93xpaO+FaB7t251zreNOuZ66tdfqwpK39jcfeWUEDQCABDQCBBDQABNJHVKR3mDqlvtK/32Qrfx7T3k/+vdgWK2gACCSgASCQgAaAQPbirujd2aR1UsCf0+myJCtoAAgkoAEgkIAGgED6E1iRv0nIMvrfnOjMt8UKGgACCWgACCSgASCQOWhYUevezWn38yabznnbrKABIJCABoBAAhoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgD/wLVtkvnbRmK44AAAAASUVORK5CYII=",
    }),
    [],
  );

  const effectiveMapState = state.map ?? FALLBACK_MAP;

  const mapAdditionalRobotMarkers = useMemo((): MapAdditionalRobotMarker[] | undefined => {
    // When a real robot is connected, the live `state.robot_pose` already
    // drives the primary robot marker — additional fleet markers (left over
    // from a Create-Operator demo flow stashed in localStorage) would render
    // as ghost duplicates. Suppress them so the map shows exactly one robot.
    if (state.connected) {
      return undefined;
    }
    if (deployedOperators.length === 0) {
      return undefined;
    }
    return deployedOperators.map((entry, index) => ({
      id: entry.id,
      card: robotOperatorHoverCardFromPolarisFleetEntry(entry),
      deployPop: robotMarkerDeployPop && index === 0,
    }));
  }, [state.connected, deployedOperators, robotMarkerDeployPop]);

  const loadAgentTools = useCallback(async () => {
    setAgentToolsLoading(true);
    setAgentToolsError(null);
    try {
      const manifest = await fetchJson<ChatToolDefinition[]>("/api/chat/tools");
      startTransition(() => {
        setAgentTools(manifest);
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load agent tools.";
      setAgentToolsError(message);
      reportActionError("Loading agent tools failed", error);
    } finally {
      setAgentToolsLoading(false);
    }
  }, [reportActionError]);

  const handleOpenAgentTools = useCallback(() => {
    setControlsMenuOpen(false);
    setAgentToolsOpen(true);
    if (!agentToolsLoading && agentTools === null && agentToolsError === null) {
      void loadAgentTools();
    }
  }, [agentTools, agentToolsError, agentToolsLoading, loadAgentTools]);

  useEffect(() => {
    if (!controlsMenuOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent): void => {
      if (!controlsMenuRef.current?.contains(event.target as Node)) {
        setControlsMenuOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setControlsMenuOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [controlsMenuOpen]);

  const handleToggleTeleop = useCallback(() => {
    teleopKeysRef.current.clear();
    setTeleopEnabled((current) => {
      const next = !current;
      appendActivity(
        "operator",
        next ? "Teleop enabled" : "Teleop disabled",
        next ? "Keyboard control armed." : "Keyboard control released.",
        next ? "accent" : "neutral",
      );
      return next;
    });
  }, [appendActivity]);

  useEffect(() => {
    teleopKeysRef.current.clear();
    teleopErrorMessageRef.current = null;

    if (!teleopEnabled) {
      void handleStopMotion();
      return undefined;
    }

    const sendCurrentCommand = async (): Promise<void> => {
      if (teleopRequestInFlightRef.current) {
        return;
      }
      teleopRequestInFlightRef.current = true;
      try {
        await fetchJson("/api/teleop/command", {
          method: "POST",
          body: JSON.stringify(calculateTeleopCommand(teleopKeysRef.current)),
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Teleop request failed";
        if (teleopErrorMessageRef.current !== message) {
          teleopErrorMessageRef.current = message;
          reportActionError("Teleop command failed", error);
          appendActivity(
            "system",
            "Teleop disabled",
            "Control path lost.",
            "danger",
          );
        }
        setTeleopEnabled(false);
      } finally {
        teleopRequestInFlightRef.current = false;
      }
    };

    const handleKeyDown = (event: KeyboardEvent): void => {
      const normalizedKey = normalizeTeleopKey(event.key);
      if (!teleopKeys.has(normalizedKey) || isEditableTarget(event.target)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      teleopKeysRef.current.add(normalizedKey);
      if (normalizedKey === " ") {
        void handleStopMotion();
      }
    };

    const handleKeyUp = (event: KeyboardEvent): void => {
      const normalizedKey = normalizeTeleopKey(event.key);
      if (!teleopKeys.has(normalizedKey) || isEditableTarget(event.target)) {
        return;
      }
      teleopKeysRef.current.delete(normalizedKey);
    };

    const handleBlur = (): void => {
      teleopKeysRef.current.clear();
      setTeleopEnabled(false);
    };

    const handleFocus = (): void => {
      teleopKeysRef.current.clear();
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);
    window.addEventListener("focus", handleFocus);
    teleopIntervalRef.current = window.setInterval(() => {
      void sendCurrentCommand();
    }, 1000 / PUBLISH_RATE_HZ);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
      window.removeEventListener("focus", handleFocus);
      if (teleopIntervalRef.current !== null) {
        window.clearInterval(teleopIntervalRef.current);
        teleopIntervalRef.current = null;
      }
      teleopKeysRef.current.clear();
      void handleStopMotion();
    };
  }, [appendActivity, handleStopMotion, reportActionError, teleopEnabled]);

  const navigatorMapHeaderActions = (
    <div className="topbar-actions polaris-navigator-map-header-actions">
      <div className="polaris-navigator-map-header-controls-group">
        <div className="topbar-menu polaris-navigator-map-settings-anchor" ref={controlsMenuRef}>
          <button
            aria-expanded={controlsMenuOpen}
            aria-haspopup="menu"
            aria-label="Settings"
            className="action-button secondary settings-cog-button polaris-navigator-map-settings-cog"
            onClick={() => {
              setControlsMenuOpen((current) => !current);
            }}
            title="Settings"
            type="button"
          >
            <SettingsCogGlyphs />
          </button>

          {controlsMenuOpen ? (
            <div className="menu-popover">
            <label className="menu-field">
              <span>Inspect mode</span>
              <select
                onChange={(event) => {
                  void handleInspectionModeChange(
                    event.target.value as ManualInspectionMode,
                  );
                  setControlsMenuOpen(false);
                }}
                value={state.inspection_settings.manual_mode}
              >
                <option value="ai_gate">AI Gate</option>
                <option value="always_create">Always Create</option>
              </select>
            </label>
            <button
              className="menu-item"
              onClick={() => {
                handleOpenAgentTools();
              }}
              type="button"
            >
              Orchestrator tool calls
            </button>
            <button
              className="menu-item"
              onClick={() => {
                void handleYoloModeChange(
                  state.yolo_runtime.mode === "live" ? "paused" : "live",
                );
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              {state.yolo_runtime.mode === "live"
                ? "Pause YOLO labeling"
                : "Resume YOLO labeling"}
            </button>
            <button
              className="menu-item"
              onClick={() => {
                void handleYoloInferenceEnabledChange(
                  !state.yolo_runtime.inference_enabled,
                );
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              {state.yolo_runtime.inference_enabled
                ? "Turn YOLO inference off"
                : "Turn YOLO inference on"}
            </button>
            <button
              className="menu-item"
              onClick={() => {
                void handleLayerToggle(
                  "show_yolo",
                  !state.layers.show_yolo,
                );
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              {state.layers.show_yolo
                ? "Hide YOLO layer"
                : "Show YOLO layer"}
            </button>
            <button
              className="menu-item"
              onClick={() => {
                void handleLayerToggle(
                  "show_pois",
                  !state.layers.show_pois,
                );
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              {state.layers.show_pois ? "Hide VLM layer" : "Show VLM layer"}
            </button>
            <div className="menu-divider" />
            <p className="menu-section-label">Danger zone</p>
            <button
              className="menu-item menu-item-danger"
              disabled={busyAction !== null || state.map === null}
              onClick={() => {
                void handleClearLowLevelMapMemory();
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              Clear low-level map
            </button>
            <button
              className="menu-item menu-item-danger"
              disabled={
                busyAction !== null ||
                (state.pois.length === 0 && state.yolo_objects.length === 0)
              }
              onClick={() => {
                void handleClearSemanticMemory();
                setControlsMenuOpen(false);
              }}
              type="button"
            >
              Clear semantic memory
            </button>
          </div>
          ) : null}
        </div>

        <NavigatorMapControlsHover
          busyAction={busyAction}
          connected={state.connected}
          inspectionRunning={state.inspection.status === "running"}
          map={state.map}
          openaiConfigured={state.openai_configured}
          robotPose={state.robot_pose}
          teleopEnabled={teleopEnabled}
          onClearFocus={() => {
            void handleClearFocus();
          }}
          onFitMap={() => {
            void handleFocusMap();
          }}
          onFocusRobot={() => {
            void handleFocusRobot();
          }}
          onInspect={() => {
            void handleInspectNow();
          }}
          onSaveMap={() => {
            void handleSaveMap();
          }}
          onStopMotion={() => {
            void handleStopMotion();
          }}
          onStopStack={() => {
            void handleStopDimos();
          }}
          onToggleTeleop={handleToggleTeleop}
        />
      </div>
    </div>
  );

  return (
    <div className="app-shell--light polaris-navigator-root">
      <div className="polaris-fade-stagger polaris-fade-stagger--navigator-workspace">
        <div className="polaris-navigator-fade-toolbar px-4 pb-1 pt-0 sm:px-8 sm:pb-1.5 sm:pt-1">
          <div className="polaris-navigator-toolbar polaris-navigator-toolbar--actions mx-auto flex w-full max-w-[min(100vw-2rem,1600px)] flex-wrap items-center gap-3">
            <div className="topbar-status">
              {slamassApiStatus === "loading" ? (
                <span className="toolbar-chip">API…</span>
              ) : null}
              {slamassApiStatus === "error" ? (
                <span className="toolbar-chip tone-danger">API unreachable</span>
              ) : null}
              {teleopEnabled ? (
                <span className="toolbar-chip tone-danger">Teleop armed</span>
              ) : null}
              {state.chat.running ? (
                <span className="toolbar-chip tone-accent">Orchestrator thinking</span>
              ) : null}
              {!state.yolo_runtime.inference_enabled ? (
                <span className="toolbar-chip tone-danger">YOLO off</span>
              ) : null}
              {state.yolo_runtime.mode !== "live" ? (
                <span className="toolbar-chip tone-accent">YOLO paused</span>
              ) : null}
            </div>
          </div>
        </div>

        <main
          className="polaris-navigator-main-split polaris-navigator-workspace"
          data-testid="polaris-navigator-main"
        >
          <aside
            aria-label="Navigator options"
            className="polaris-navigator-sidebar"
          >
            <div className="polaris-navigator-operations-column polaris-navigator-map">
              <div className="polaris-navigator-operations-inner polaris-navigator-operations-body">
                <NavigatorOperatorFleet
                  hideDemoFleet
                  onGo2OperatorHoverChange={setGo2OperatorFleetHover}
                  prependedOperators={
                    state.connected
                      ? [
                          {
                            id: "live-go2",
                            title: "Unitree Go2",
                            category: { label: "Type", value: "Unitree Go2" },
                            location: "Live link",
                            task: "Awaiting command",
                            batteryPercent: state.battery_percent,
                            active: "green",
                            imageUrl: POLARIS_GO2_PREVIEW_URL,
                            imageAlt: "Connected Unitree Go2",
                            mountThumbUrl: POLARIS_OPERATOR_SELECT_THUMB_URL,
                          },
                        ]
                      : []
                  }
                />

                <NavigatorTeleopPanel connected={state.connected} />

                <NavigatorOptionCard
                  bodyVariant="scroll"
                  className="polaris-nav-option-card--grow polaris-nav-option-card--polaris-heading"
                  title="Orchestrator"
                >
                  <OperatorRail
                    activityEntries={activityEntries}
                    busyAction={busyAction}
                    chat={state.chat}
                    embedSegment="agent"
                    embedded
                    items={semanticItems}
                    onClearFocus={() => {
                      void handleClearFocus();
                    }}
                    onFocusItem={(item) => {
                      void handleFocusItem(item);
                    }}
                    onGoToItem={(item) => {
                      void handleGoToItem(item);
                    }}
                    onHighlightItem={(item) => {
                      void handleHighlightItem(item);
                    }}
                    onResetChat={() => {
                      void handleResetChat();
                    }}
                    onSelectItem={(item) => {
                      void handleSelectItem(item);
                    }}
                    onSubmitChatMessage={(message) => {
                      void handleSubmitChatMessage(message);
                    }}
                    selectedItem={state.ui.selected_item}
                    selectedPreview={selectedPreview}
                  />
                </NavigatorOptionCard>
              </div>
            </div>
          </aside>

          <div className="polaris-navigator-map-column polaris-navigator-map">
            <PanelShell
              aside={navigatorMapHeaderActions}
              bodyClassName="panel-body-stage"
              className="map-panel polaris-navigator-map-panel--compact-title"
              title="Navigator"
            >
              <MapPane
                additionalRobotMarkers={mapAdditionalRobotMarkers}
                layers={state.layers}
                map={effectiveMapState}
                operatorFleetGo2Hover={go2OperatorFleetHover}
                pinViewModeControlsBottom
                refitOnLayoutReady
                viewportScreenAnchorY={NAVIGATOR_MAP_VIEWPORT_ANCHOR_Y}
                robotOperatorHoverCard={defaultRobotOperatorHoverCard("navigator")}
                showMapToolbar={false}
                onCameraChange={queueCameraSync}
                onClearFocus={() => {
                  void handleClearFocus();
                }}
                onFocusItem={(item) => {
                  void handleFocusItem(item);
                }}
                onFocusMap={() => {
                  void handleFocusMap();
                }}
                onFocusRobot={() => {
                  void handleFocusRobot();
                }}
                onNavigate={(x, y) => {
                  void handleNavigate(x, y);
                }}
                onSelectItem={(item) => {
                  void handleSelectItem(item);
                }}
                path={state.path}
                pois={state.pois}
                robotPose={state.robot_pose}
                ui={state.ui}
                yoloObjects={state.yolo_objects}
              />
            </PanelShell>

            <div
              aria-label="Latest activity"
              className="polaris-navigator-map-activity-below"
              data-testid="polaris-navigator-activity-line"
            >
              <p
                aria-live="polite"
                className={`polaris-navigator-activity-line${latestActivity ? ` ${navigatorActivityToneClass(latestActivity.tone)}` : " polaris-navigator-activity-line--neutral"}`}
              >
                {latestActivity
                  ? formatNavigatorActivityLine(latestActivity)
                  : "No recent activity"}
              </p>
            </div>
          </div>

          <aside
            aria-label="Camera and detections"
            className="polaris-navigator-detections-column polaris-navigator-map"
          >
            <NavigatorOptionCard
              className="polaris-nav-detections-camera-card polaris-nav-option-card--polaris-heading"
              headerAside={
                <button
                  aria-label={
                    cameraHeaderCaptureEnabled
                      ? state.connected
                        ? "Take a picture (live feed)"
                        : "Take a picture"
                      : "Camera feed not ready"
                  }
                  className="polaris-nav-camera-header-capture"
                  disabled={!cameraHeaderCaptureEnabled}
                  type="button"
                  onClick={() => {
                    cameraFeedRef.current?.captureSnapshot();
                  }}
                >
                  <CameraIcon
                    aria-hidden
                    className="polaris-nav-camera-header-capture-icon"
                  />
                </button>
              }
              title="Camera"
            >
              <LiveFeedPanel
                ref={cameraFeedRef}
                connected={state.connected}
                embedded
                frameLabel=""
                onCaptureAvailabilityChange={setCameraHeaderCaptureEnabled}
                poseLabel={null}
                pov={state.pov}
              />
            </NavigatorOptionCard>

            <NavigatorOptionCard
              bodyClassName="polaris-navigator-detections-body"
              bodyVariant="scroll"
              className="polaris-nav-detections-list-card polaris-nav-option-card--grow polaris-nav-option-card--polaris-heading"
              title="Detection log"
            >
              <OperatorRail
                activityEntries={activityEntries}
                busyAction={busyAction}
                chat={state.chat}
                embedSegment="memory"
                embedded
                items={semanticItems}
                onClearFocus={() => {
                  void handleClearFocus();
                }}
                onFocusItem={(item) => {
                  void handleFocusItem(item);
                }}
                onGoToItem={(item) => {
                  void handleGoToItem(item);
                }}
                onHighlightItem={(item) => {
                  void handleHighlightItem(item);
                }}
                onResetChat={() => {
                  void handleResetChat();
                }}
                onSelectItem={(item) => {
                  void handleSelectItem(item);
                }}
                onSubmitChatMessage={(message) => {
                  void handleSubmitChatMessage(message);
                }}
                selectedItem={state.ui.selected_item}
                selectedPreview={selectedPreview}
              />
            </NavigatorOptionCard>
          </aside>
        </main>
      </div>

      {selectedPoi || selectedYoloObject ? (
        <div
          className="poi-modal-backdrop"
          onClick={() => {
            void handleSelectItem(null);
          }}
        >
          <div
            className="poi-modal"
            onClick={(event) => event.stopPropagation()}
          >
            {selectedPoi ? (
              <>
                <div className="poi-modal-media">
                  <img
                    alt={selectedPoi.title}
                    src={selectedPoi.hero_image_url}
                  />
                </div>
                <div className="poi-modal-body">
                  <div className="poi-modal-header">
                    <div>
                      <p className="eyebrow">{selectedPoi.category}</p>
                      <h3>{selectedPoi.title}</h3>
                    </div>
                    <div className="poi-modal-header-actions">
                      <span className="score-pill">
                        {selectedPoi.interest_score.toFixed(2)}
                      </span>
                      <button
                        className="close-button"
                        onClick={() => {
                          void handleSelectItem(null);
                        }}
                        type="button"
                      >
                        Close
                      </button>
                    </div>
                  </div>

                  <p className="poi-summary">{selectedPoi.summary}</p>

                  <div className="poi-meta">
                    <span>
                      Target {selectedPoi.target_x.toFixed(2)},{" "}
                      {selectedPoi.target_y.toFixed(2)}
                    </span>
                    <span>
                      View {selectedPoi.anchor_x.toFixed(2)},{" "}
                      {selectedPoi.anchor_y.toFixed(2)} |{" "}
                      {formatYaw(selectedPoi.anchor_yaw)}
                    </span>
                    <span>{formatTimestamp(selectedPoi.updated_at)}</span>
                  </div>

                  {selectedPoi.objects.length > 0 ? (
                    <div className="poi-tags">
                      {selectedPoi.objects.map((item) => (
                        <span key={item}>{item}</span>
                      ))}
                    </div>
                  ) : null}

                  <div className="poi-actions">
                    <button
                      className="action-button secondary"
                      onClick={() => {
                        void handleHighlightItem({
                          kind: "vlm_poi",
                          entity_id: selectedPoi.poi_id,
                        });
                      }}
                      type="button"
                    >
                      Highlight
                    </button>
                    <button
                      className="action-button secondary"
                      onClick={() => {
                        void handleFocusItem({
                          kind: "vlm_poi",
                          entity_id: selectedPoi.poi_id,
                        });
                      }}
                      type="button"
                    >
                      Focus
                    </button>
                    <button
                      className="action-button"
                      disabled={
                        busyAction === `go-vlm_poi-${selectedPoi.poi_id}`
                      }
                      onClick={() => {
                        void handleGoToItem({
                          kind: "vlm_poi",
                          entity_id: selectedPoi.poi_id,
                        });
                      }}
                      type="button"
                    >
                      Go To
                    </button>
                    <button
                      className="action-button danger"
                      disabled={
                        busyAction === `delete-vlm_poi-${selectedPoi.poi_id}`
                      }
                      onClick={() => {
                        void handleDeleteItem({
                          kind: "vlm_poi",
                          entity_id: selectedPoi.poi_id,
                        });
                      }}
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </>
            ) : selectedYoloObject ? (
              <>
                <div className="poi-modal-media">
                  <img
                    alt={selectedYoloObject.label}
                    src={selectedYoloObject.hero_image_url}
                  />
                </div>
                <div className="poi-modal-body">
                  <div className="poi-modal-header">
                    <div>
                      <p className="eyebrow">YOLO object</p>
                      <h3>{selectedYoloObject.label}</h3>
                    </div>
                    <div className="poi-modal-header-actions">
                      <span className="score-pill">
                        {Math.round(selectedYoloObject.best_confidence * 100)}%
                      </span>
                      <button
                        className="close-button"
                        onClick={() => {
                          void handleSelectItem(null);
                        }}
                        type="button"
                      >
                        Close
                      </button>
                    </div>
                  </div>

                  <p className="poi-summary">
                    Stable YOLO object promoted from{" "}
                    {selectedYoloObject.detections_count} detections. Navigation
                    will restore the best recorded viewing pose.
                  </p>

                  <div className="poi-meta">
                    <span>
                      Object {selectedYoloObject.world_x.toFixed(2)},{" "}
                      {selectedYoloObject.world_y.toFixed(2)}
                    </span>
                    <span>
                      View {selectedYoloObject.best_view_x.toFixed(2)},{" "}
                      {selectedYoloObject.best_view_y.toFixed(2)} |{" "}
                      {formatYaw(selectedYoloObject.best_view_yaw)}
                    </span>
                    <span>
                      {formatTimestamp(selectedYoloObject.last_seen_at)}
                    </span>
                  </div>

                  <div className="poi-tags">
                    <span>{selectedYoloObject.label}</span>
                    <span>{selectedYoloObject.detections_count} hits</span>
                    <span>
                      {selectedYoloObject.size_x.toFixed(2)}m ×{" "}
                      {selectedYoloObject.size_y.toFixed(2)}m
                    </span>
                  </div>

                  <div className="poi-actions">
                    <button
                      className="action-button secondary"
                      onClick={() => {
                        void handleHighlightItem({
                          kind: "yolo_object",
                          entity_id: selectedYoloObject.object_id,
                        });
                      }}
                      type="button"
                    >
                      Highlight
                    </button>
                    <button
                      className="action-button secondary"
                      onClick={() => {
                        void handleFocusItem({
                          kind: "yolo_object",
                          entity_id: selectedYoloObject.object_id,
                        });
                      }}
                      type="button"
                    >
                      Focus
                    </button>
                    <button
                      className="action-button"
                      disabled={
                        busyAction ===
                        `go-yolo_object-${selectedYoloObject.object_id}`
                      }
                      onClick={() => {
                        void handleGoToItem({
                          kind: "yolo_object",
                          entity_id: selectedYoloObject.object_id,
                        });
                      }}
                      type="button"
                    >
                      Go To
                    </button>
                    <button
                      className="action-button danger"
                      disabled={
                        busyAction ===
                        `delete-yolo_object-${selectedYoloObject.object_id}`
                      }
                      onClick={() => {
                        void handleDeleteItem({
                          kind: "yolo_object",
                          entity_id: selectedYoloObject.object_id,
                        });
                      }}
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}

      {agentToolsOpen ? (
        <AgentToolsModal
          error={agentToolsError}
          loading={agentToolsLoading}
          onClose={() => {
            setAgentToolsOpen(false);
          }}
          onReload={() => {
            void loadAgentTools();
          }}
          tools={agentTools}
        />
      ) : null}
    </div>
  );
}
