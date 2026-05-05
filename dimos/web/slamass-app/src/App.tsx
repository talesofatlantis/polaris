import React, { startTransition } from "react";

import { apiUrl, normalizeAppStateForDev } from "./apiBase";
import { AgentToolsModal } from "./AgentToolsModal";
import { LiveFeedPanel } from "./LiveFeedPanel";
import { MapPane } from "./MapPane";
import { defaultRobotOperatorHoverCard } from "./robotOperatorLabel";
import { OperatorRail, SelectedSemanticPreview } from "./OperatorRail";
import { PanelShell } from "./PanelShell";
import { SaveMapGlyphs } from "./SaveMapGlyphs";
import { SettingsCogGlyphs } from "./SettingsCogGlyphs";
import {
  buildSemanticItems,
  isPolarisDemoCameraCaptureRef,
  mergePoi,
  mergeYoloObject,
  polarisDemoCameraCapturePreviewFields,
  resolveSelectedPoi,
  resolveSelectedYoloObject,
} from "./semanticItems";
import {
  calculateTeleopCommand,
  isEditableTarget,
  normalizeTeleopKey,
  PUBLISH_RATE_HZ,
  teleopKeys,
} from "./teleop";
import { applyFitOverviewCameraToAppState, DEFAULT_MAP_ZOOM } from "./mapViewport";
import { mergeUiPreferringLocalCamera } from "./uiStateMerge";
import {
  AppState,
  ChatState,
  ChatToolDefinition,
  InspectionSettings,
  ManualInspectionMode,
  Poi,
  SemanticItemRef,
  UiCameraState,
  UiState,
  YoloObject,
  YoloRuntimeMode,
} from "./types";

const LAYOUT_STORAGE_KEY = "slamass-layout-mode";
const MAX_ACTIVITY_ENTRIES = 10;

const emptyState: AppState = {
  dimos_viewer_url: null,
  dimos_rerun_web_viewer_url: null,
  openai_configured: true,
  connected: false,
  battery_percent: null,
  robot_pose: null,
  path: [],
  pov: {
    available: false,
    seq: 0,
    updated_at: null,
    image_url: apiUrl("/api/pov/latest.jpg?v=0"),
  },
  map: null,
  pois: [],
  yolo_objects: [],
  inspection: {
    status: "idle",
    message: "",
    poi_id: null,
  },
  inspection_settings: {
    manual_mode: "ai_gate",
  },
  yolo_runtime: {
    mode: "live",
    inference_enabled: true,
  },
  layers: {
    show_pois: true,
    show_yolo: true,
  },
  ui: {
    revision: 0,
    camera: {
      center_x: null,
      center_y: null,
      zoom: DEFAULT_MAP_ZOOM,
    },
    selected_item: null,
    highlighted_items: [],
  },
  chat: {
    running: false,
    messages: [],
  },
};

type LayoutMode = "duo" | "trio";
type ActivityRole = "operator" | "system";
type ActivityTone = "neutral" | "accent" | "success" | "danger";
type ActivityEntry = {
  id: string;
  role: ActivityRole;
  tone: ActivityTone;
  title: string;
  detail: string;
  timestamp: string;
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(url), {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status}`);
  }
  return (await response.json()) as T;
}

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

function formatClock(date: Date): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatPoseLabel(state: AppState): string | null {
  if (!state.robot_pose) {
    return null;
  }
  return `${state.robot_pose.x.toFixed(2)}, ${state.robot_pose.y.toFixed(2)} | ${formatYaw(
    state.robot_pose.yaw,
  )}`;
}

function applyUiState(previous: UiState, next: UiState): UiState {
  return next.revision >= previous.revision ? next : previous;
}

function inspectionSignature(inspection: AppState["inspection"]): string {
  return `${inspection.status}|${inspection.message}|${inspection.poi_id ?? ""}`;
}

function getInitialLayoutMode(): LayoutMode {
  if (typeof window === "undefined") {
    return "duo";
  }
  const stored = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
  if (stored === "duo" || stored === "trio") {
    return stored;
  }
  return window.innerWidth >= 1280 ? "trio" : "duo";
}

function persistLayoutMode(next: LayoutMode): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(LAYOUT_STORAGE_KEY, next);
}

function toneForInspection(status: string): ActivityTone {
  if (status === "accepted") {
    return "success";
  }
  if (status === "failed") {
    return "danger";
  }
  if (status === "running" || status === "rejected") {
    return "accent";
  }
  return "neutral";
}

function describeSemanticItem(item: SemanticItemRef): string {
  return item.kind === "vlm_poi" ? "POI" : "YOLO object";
}

export default function App(): React.ReactElement {
  const [state, setState] = React.useState<AppState>(emptyState);
  const [busyAction, setBusyAction] = React.useState<string | null>(null);
  const [layoutMode, setLayoutModeState] = React.useState<LayoutMode>(getInitialLayoutMode);
  const [teleopEnabled, setTeleopEnabled] = React.useState(false);
  const [controlsMenuOpen, setControlsMenuOpen] = React.useState(false);
  const [agentToolsOpen, setAgentToolsOpen] = React.useState(false);
  const [agentToolsLoading, setAgentToolsLoading] = React.useState(false);
  const [agentToolsError, setAgentToolsError] = React.useState<string | null>(null);
  const [agentTools, setAgentTools] = React.useState<ChatToolDefinition[] | null>(null);
  const [activityEntries, setActivityEntries] = React.useState<ActivityEntry[]>(() => [
    {
      id: "boot",
      role: "system",
      tone: "neutral",
      title: "Session started",
      detail: "Waiting for POV and map updates.",
      timestamp: formatClock(new Date()),
    },
  ]);

  const cameraSyncTimerRef = React.useRef<number | null>(null);
  const pendingLocalCameraRef = React.useRef<UiCameraState | null>(null);
  const cameraPutInFlightRef = React.useRef(false);
  const teleopIntervalRef = React.useRef<number | null>(null);
  const teleopKeysRef = React.useRef<Set<string>>(new Set());
  const teleopRequestInFlightRef = React.useRef(false);
  const teleopErrorMessageRef = React.useRef<string | null>(null);
  const controlsMenuRef = React.useRef<HTMLDivElement>(null);
  const activityCounterRef = React.useRef(1);
  const stateRef = React.useRef<AppState>(emptyState);
  const lastConnectedRef = React.useRef<boolean | null>(null);
  const lastInspectionRef = React.useRef<string>("");
  const mapReadyRef = React.useRef(false);
  const didLogInitialSnapshotRef = React.useRef(false);
  const initialFetchErrorLoggedRef = React.useRef(false);
  /** One fit + focus-map sync per page load; reset when map clears (see map_updated). */
  const sessionMapFitDoneRef = React.useRef(false);

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

  const appendActivity = React.useCallback(
    (role: ActivityRole, title: string, detail: string, tone: ActivityTone = "neutral") => {
      const entry = {
        id: `activity-${activityCounterRef.current}`,
        role,
        tone,
        title,
        detail,
        timestamp: formatClock(new Date()),
      };
      activityCounterRef.current += 1;
      setActivityEntries((previous) => [...previous, entry].slice(-MAX_ACTIVITY_ENTRIES));
    },
    [],
  );

  const reportActionError = React.useCallback(
    (title: string, error: unknown) => {
      const message = error instanceof Error ? error.message : "Request failed";
      appendActivity("system", title, message, "danger");
    },
    [appendActivity],
  );

  React.useEffect(() => {
    stateRef.current = state;
  }, [state]);

  React.useEffect(() => {
    return () => {
      if (cameraSyncTimerRef.current !== null) {
        window.clearTimeout(cameraSyncTimerRef.current);
      }
      if (teleopIntervalRef.current !== null) {
        window.clearInterval(teleopIntervalRef.current);
      }
    };
  }, []);

  React.useEffect(() => {
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

  const setLayoutMode = React.useCallback((next: LayoutMode) => {
    setLayoutModeState(next);
    persistLayoutMode(next);
  }, []);

  const mergeUiState = React.useCallback((nextUi: UiState) => {
    startTransition(() => {
      setState((previous) => ({
        ...previous,
        ui: applyUiState(previous.ui, nextUi),
      }));
    });
  }, []);

  const issueUiCommand = React.useCallback(
    async (url: string, init?: RequestInit): Promise<UiState> => {
      if (cameraSyncTimerRef.current !== null) {
        window.clearTimeout(cameraSyncTimerRef.current);
        cameraSyncTimerRef.current = null;
      }
      pendingLocalCameraRef.current = null;
      cameraPutInFlightRef.current = false;
      const nextUi = await fetchJson<UiState>(url, init);
      mergeUiState(nextUi);
      return nextUi;
    },
    [mergeUiState],
  );

  const queueCameraSync = React.useCallback(
    (camera: UiCameraState) => {
      pendingLocalCameraRef.current = camera;
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          ui: {
            ...previous.ui,
            camera,
          },
        }));
      });

      if (cameraSyncTimerRef.current !== null) {
        window.clearTimeout(cameraSyncTimerRef.current);
      }
      cameraSyncTimerRef.current = window.setTimeout(() => {
        cameraSyncTimerRef.current = null;
        const snapshot = pendingLocalCameraRef.current;
        if (snapshot === null) {
          return;
        }
        cameraPutInFlightRef.current = true;
        void (async () => {
          try {
            const nextUi = await fetchJson<UiState>("/api/ui/camera", {
              method: "PUT",
              body: JSON.stringify(snapshot),
            });
            pendingLocalCameraRef.current = null;
            mergeUiState(nextUi);
          } catch (error) {
            reportActionError("Camera sync failed", error);
          } finally {
            cameraPutInFlightRef.current = false;
          }
        })();
      }, 140);
    },
    [mergeUiState, reportActionError],
  );

  React.useEffect(() => {
    let cancelled = false;

    const loadStateOnce = async (): Promise<boolean> => {
      try {
        const data = await fetchJson<AppState>("/api/state");
        if (cancelled) {
          return true;
        }

        lastConnectedRef.current = data.connected;
        lastInspectionRef.current = inspectionSignature(data.inspection);
        mapReadyRef.current = data.map !== null;

        if (!didLogInitialSnapshotRef.current) {
          didLogInitialSnapshotRef.current = true;
          appendActivity(
            "system",
            data.connected ? "Robot online" : "Robot offline",
            data.map ? "Map loaded." : "Waiting for map data.",
            data.connected ? "success" : "neutral",
          );
        }

        const normalized = normalizeAppStateForDev(data);
        const withFit = data.map ? applyFitOverviewCameraToAppState(normalized) : normalized;
        let postFocusMapFromLoad = false;
        if (data.map && !sessionMapFitDoneRef.current) {
          sessionMapFitDoneRef.current = true;
          postFocusMapFromLoad = true;
        }

        startTransition(() => {
          setState(withFit);
        });

        if (postFocusMapFromLoad && !cancelled) {
          void fetchJson<UiState>("/api/ui/focus-map", { method: "POST" })
            .then((ui) => {
              if (cancelled) {
                return;
              }
              mergeUiState(ui);
            })
            .catch(() => {
              /* Map may disappear before the call completes. */
            });
        }
        return true;
      } catch (error) {
        if (!initialFetchErrorLoggedRef.current) {
          initialFetchErrorLoggedRef.current = true;
          reportActionError("Navigator API unreachable; retrying…", error);
        }
        return false;
      }
    };

    const runInitialLoad = async (): Promise<void> => {
      let attempt = 0;
      while (!cancelled) {
        const ok = await loadStateOnce();
        if (ok || cancelled) {
          return;
        }
        attempt += 1;
        const delayMs = Math.min(2_500 + attempt * 400, 12_000);
        await new Promise((resolve) => {
          window.setTimeout(resolve, delayMs);
        });
      }
    };

    void runInitialLoad();

    const source = new EventSource(apiUrl("/api/events"));

    source.addEventListener("state_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as Partial<AppState>;

      if (typeof payload.connected === "boolean" && payload.connected !== lastConnectedRef.current) {
        lastConnectedRef.current = payload.connected;
        appendActivity(
          "system",
          payload.connected ? "Robot reconnected" : "Robot disconnected",
          payload.connected ? "Live socket restored." : "Holding last known state.",
          payload.connected ? "success" : "danger",
        );
      }

      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            ...payload,
            pov: payload.pov ? { ...previous.pov, ...payload.pov } : previous.pov,
            yolo_runtime: payload.yolo_runtime ?? previous.yolo_runtime,
            layers: payload.layers ?? previous.layers,
            inspection_settings: payload.inspection_settings ?? previous.inspection_settings,
          }),
        );
      });
    });

    source.addEventListener("map_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as AppState["map"];
      if (payload === null) {
        sessionMapFitDoneRef.current = false;
      }
      if (!mapReadyRef.current && payload) {
        mapReadyRef.current = true;
        appendActivity("system", "Map ready", "Occupancy map is rendering.", "success");
      }
      const shouldSessionFit = Boolean(payload) && !sessionMapFitDoneRef.current;
      if (shouldSessionFit) {
        sessionMapFitDoneRef.current = true;
      }
      startTransition(() => {
        setState((previous) => {
          const next = normalizeAppStateForDev({ ...previous, map: payload });
          return shouldSessionFit ? applyFitOverviewCameraToAppState(next) : next;
        });
      });
      if (shouldSessionFit && !cancelled) {
        void fetchJson<UiState>("/api/ui/focus-map", { method: "POST" })
          .then((ui) => {
            if (cancelled) {
              return;
            }
            mergeUiState(ui);
          })
          .catch(() => {
            /* ignore */
          });
      }
    });

    source.addEventListener("poi_upserted", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as Poi;
      const existed = stateRef.current.pois.some((poi) => poi.poi_id === payload.poi_id);
      appendActivity("system", existed ? "POI updated" : "POI added", payload.title, "success");
      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            pois: mergePoi(previous.pois, payload),
          }),
        );
      });
    });

    source.addEventListener("poi_deleted", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as { poi_id: string };
      const deletedPoi = stateRef.current.pois.find((poi) => poi.poi_id === payload.poi_id);
      appendActivity(
        "system",
        "POI deleted",
        deletedPoi ? deletedPoi.title : "Semantic anchor removed.",
        "danger",
      );
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          pois: previous.pois.filter((poi) => poi.poi_id !== payload.poi_id),
        }));
      });
    });

    source.addEventListener("yolo_object_upserted", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as YoloObject;
      const existed = stateRef.current.yolo_objects.some(
        (object) => object.object_id === payload.object_id,
      );
      appendActivity(
        "system",
        existed ? "YOLO object updated" : "YOLO object promoted",
        payload.label,
        "success",
      );
      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            yolo_objects: mergeYoloObject(previous.yolo_objects, payload),
          }),
        );
      });
    });

    source.addEventListener("yolo_object_deleted", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as { object_id: string };
      const deletedObject = stateRef.current.yolo_objects.find(
        (object) => object.object_id === payload.object_id,
      );
      appendActivity(
        "system",
        "YOLO object deleted",
        deletedObject ? deletedObject.label : "Tracked object removed.",
        "danger",
      );
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          yolo_objects: previous.yolo_objects.filter(
            (object) => object.object_id !== payload.object_id,
          ),
        }));
      });
    });

    source.addEventListener("inspection_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as AppState["inspection"];
      const signature = inspectionSignature(payload);
      if (signature !== lastInspectionRef.current) {
        lastInspectionRef.current = signature;
        appendActivity(
          "system",
          `Inspection ${payload.status}`,
          payload.message || "Inspection state changed.",
          toneForInspection(payload.status),
        );
      }
      startTransition(() => {
        setState((previous) => ({ ...previous, inspection: payload }));
      });
    });

    source.addEventListener("chat_state_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as ChatState;
      startTransition(() => {
        setState((previous) => ({ ...previous, chat: payload }));
      });
    });

    source.addEventListener("ui_state_updated", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as UiState;
      startTransition(() => {
        setState((previous) => {
          const merged = applyUiState(previous.ui, payload);
          const holdCamera =
            pendingLocalCameraRef.current !== null &&
            (cameraSyncTimerRef.current !== null || cameraPutInFlightRef.current);
          if (holdCamera) {
            return {
              ...previous,
              ui: {
                ...merged,
                camera: pendingLocalCameraRef.current as UiCameraState,
              },
            };
          }
          return { ...previous, ui: merged };
        });
      });
    });

    return () => {
      cancelled = true;
      source.close();
      if (cameraSyncTimerRef.current !== null) {
        window.clearTimeout(cameraSyncTimerRef.current);
        cameraSyncTimerRef.current = null;
      }
    };
  }, [appendActivity, mergeUiState, reportActionError]);

  const handleInspectNow = React.useCallback(async () => {
    appendActivity("operator", "Inspect", "Manual semantic inspection started.", "accent");
    setBusyAction("inspect");
    try {
      await fetchJson("/api/inspect/now", { method: "POST" });
    } catch (error) {
      reportActionError("Inspect request failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleSaveMap = React.useCallback(async () => {
    appendActivity("operator", "Save map", "Checkpoint requested.", "accent");
    setBusyAction("save");
    try {
      await fetchJson("/api/map/save", { method: "POST" });
      appendActivity("system", "Map saved", "Checkpoint written.", "success");
    } catch (error) {
      reportActionError("Save map failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleClearLowLevelMapMemory = React.useCallback(async () => {
    const confirmed = window.confirm(
      "Clear the low-level persistent map memory?\n\nThis removes the active occupancy map only. Semantic POIs and YOLO objects are kept.",
    );
    if (!confirmed) {
      return;
    }

    appendActivity("operator", "Clear map memory", "Low-level map reset requested.", "danger");
    setBusyAction("clear-map-memory");
    try {
      await fetchJson("/api/memory/clear-map", { method: "POST" });
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          map: null,
          path: [],
          ui: {
            ...previous.ui,
            camera: {
              center_x: null,
              center_y: null,
              zoom: DEFAULT_MAP_ZOOM,
            },
          },
        }));
      });
      appendActivity("system", "Map memory cleared", "Active occupancy map removed.", "danger");
    } catch (error) {
      reportActionError("Clear map memory failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleClearSemanticMemory = React.useCallback(async () => {
    const confirmed = window.confirm(
      "Clear semantic memory?\n\nThis removes VLM POIs, YOLO objects, semantic observations, and chat history. The low-level occupancy map stays intact.",
    );
    if (!confirmed) {
      return;
    }

    appendActivity("operator", "Clear semantic memory", "Semantic memory reset requested.", "danger");
    setBusyAction("clear-semantic-memory");
    try {
      await fetchJson("/api/memory/clear-semantic", { method: "POST" });
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          pois: [],
          yolo_objects: [],
          inspection: {
            status: "idle",
            message: "",
            poi_id: null,
          },
          ui: {
            ...previous.ui,
            selected_item: null,
            highlighted_items: [],
          },
          chat: {
            running: false,
            messages: [],
          },
        }));
      });
      appendActivity(
        "system",
        "Semantic memory cleared",
        "POIs, YOLO objects, and semantic chat state removed.",
        "danger",
      );
    } catch (error) {
      reportActionError("Clear semantic memory failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleInspectionModeChange = React.useCallback(
    async (manualMode: ManualInspectionMode) => {
      try {
        const nextSettings = await fetchJson<InspectionSettings>("/api/inspection-settings", {
          method: "PUT",
          body: JSON.stringify({ manual_mode: manualMode }),
        });
        startTransition(() => {
          setState((previous) => ({
            ...previous,
            inspection_settings: nextSettings,
          }));
        });
      } catch (error) {
        reportActionError("Inspection mode update failed", error);
      }
    },
    [reportActionError],
  );

  const loadAgentTools = React.useCallback(async () => {
    setAgentToolsLoading(true);
    setAgentToolsError(null);
    try {
      const manifest = await fetchJson<ChatToolDefinition[]>("/api/chat/tools");
      startTransition(() => {
        setAgentTools(manifest);
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load agent tools.";
      setAgentToolsError(message);
      reportActionError("Loading agent tools failed", error);
    } finally {
      setAgentToolsLoading(false);
    }
  }, [reportActionError]);

  const handleOpenAgentTools = React.useCallback(() => {
    setControlsMenuOpen(false);
    setAgentToolsOpen(true);
    if (!agentToolsLoading && agentTools === null && agentToolsError === null) {
      void loadAgentTools();
    }
  }, [agentTools, agentToolsError, agentToolsLoading, loadAgentTools]);

  const handleYoloModeChange = React.useCallback(
    async (mode: YoloRuntimeMode) => {
      try {
        const payload = await fetchJson<AppState["yolo_runtime"]>("/api/yolo/runtime", {
          method: "PUT",
          body: JSON.stringify({ mode }),
        });
        startTransition(() => {
          setState((previous) => ({ ...previous, yolo_runtime: payload }));
        });
      } catch (error) {
        reportActionError("YOLO runtime update failed", error);
      }
    },
    [reportActionError],
  );

  const handleYoloInferenceEnabledChange = React.useCallback(
    async (inferenceEnabled: boolean) => {
      try {
        const payload = await fetchJson<AppState["yolo_runtime"]>("/api/yolo/runtime", {
          method: "PUT",
          body: JSON.stringify({ inference_enabled: inferenceEnabled }),
        });
        startTransition(() => {
          setState((previous) => ({ ...previous, yolo_runtime: payload }));
        });
      } catch (error) {
        reportActionError("YOLO inference update failed", error);
      }
    },
    [reportActionError],
  );

  const handleLayerToggle = React.useCallback(
    async (field: "show_pois" | "show_yolo", value: boolean) => {
      try {
        const payload = await fetchJson<AppState["layers"]>("/api/layers", {
          method: "PUT",
          body: JSON.stringify({ [field]: value }),
        });
        startTransition(() => {
          setState((previous) => ({ ...previous, layers: payload }));
        });
      } catch (error) {
        reportActionError("Layer update failed", error);
      }
    },
    [reportActionError],
  );

  const handleNavigate = React.useCallback(
    async (x: number, y: number) => {
      appendActivity("operator", "Navigate", `${x.toFixed(2)}, ${y.toFixed(2)}`, "accent");
      try {
        try {
          const nextUi = await fetchJson<UiState>("/api/ui/select-item", {
            method: "POST",
            body: JSON.stringify({ kind: null, entity_id: null }),
          });
          const preserveCamera =
            pendingLocalCameraRef.current !== null ||
            cameraSyncTimerRef.current !== null ||
            cameraPutInFlightRef.current;
          startTransition(() => {
            setState((previous) => ({
              ...previous,
              ui: mergeUiPreferringLocalCamera(
                previous.ui,
                nextUi,
                preserveCamera,
              ),
            }));
          });
        } catch (selectError) {
          reportActionError("Clear map selection failed", selectError);
        }
        await fetchJson("/api/navigate", {
          method: "POST",
          body: JSON.stringify({ x, y }),
        });
      } catch (error) {
        reportActionError("Navigation request failed", error);
      }
    },
    [appendActivity, reportActionError],
  );

  const handleGoToItem = React.useCallback(
    async (item: SemanticItemRef) => {
      if (isPolarisDemoCameraCaptureRef(item)) {
        appendActivity(
          "operator",
          "Go to item",
          "Sample capture — connect the stack to navigate to real detections.",
          "neutral",
        );
        return;
      }
      const actionKey = `go-${item.kind}-${item.entity_id}`;
      appendActivity("operator", "Go to item", describeSemanticItem(item), "accent");
      setBusyAction(actionKey);
      try {
        if (item.kind === "vlm_poi") {
          await fetchJson(`/api/pois/${item.entity_id}/go`, { method: "POST" });
        } else {
          await fetchJson(`/api/yolo-objects/${item.entity_id}/go`, { method: "POST" });
        }
      } catch (error) {
        reportActionError("Go To request failed", error);
      } finally {
        setBusyAction(null);
      }
    },
    [appendActivity, reportActionError],
  );

  const handleDeleteItem = React.useCallback(
    async (item: SemanticItemRef) => {
      const confirmed = window.confirm("Delete this semantic item from the Navigator map?");
      if (!confirmed) {
        return;
      }
      const actionKey = `delete-${item.kind}-${item.entity_id}`;
      setBusyAction(actionKey);
      try {
        if (item.kind === "vlm_poi") {
          await fetchJson(`/api/pois/${item.entity_id}/delete`, { method: "POST" });
        } else {
          await fetchJson(`/api/yolo-objects/${item.entity_id}/delete`, { method: "POST" });
        }
      } catch (error) {
        reportActionError("Delete item failed", error);
      } finally {
        setBusyAction(null);
      }
    },
    [reportActionError],
  );

  const handleSelectItem = React.useCallback(
    async (item: SemanticItemRef | null) => {
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          ui: {
            ...previous.ui,
            selected_item: item,
          },
        }));
      });
      if (item != null && isPolarisDemoCameraCaptureRef(item)) {
        return;
      }
      try {
        await issueUiCommand("/api/ui/select-item", {
          method: "POST",
          body: JSON.stringify({
            kind: item?.kind ?? null,
            entity_id: item?.entity_id ?? null,
          }),
        });
      } catch (error) {
        reportActionError("Select item failed", error);
      }
    },
    [issueUiCommand, reportActionError],
  );

  const handleFocusItem = React.useCallback(
    async (item: SemanticItemRef) => {
      if (isPolarisDemoCameraCaptureRef(item)) {
        return;
      }
      try {
        await issueUiCommand(`/api/ui/focus-item/${item.kind}/${item.entity_id}`, {
          method: "POST",
          body: JSON.stringify({}),
        });
      } catch (error) {
        reportActionError("Focus item failed", error);
      }
    },
    [issueUiCommand, reportActionError],
  );

  const handleHighlightItem = React.useCallback(
    async (item: SemanticItemRef) => {
      if (isPolarisDemoCameraCaptureRef(item)) {
        return;
      }
      try {
        await issueUiCommand("/api/ui/highlight-items", {
          method: "POST",
          body: JSON.stringify({
            items: [item],
            selected_item: item,
          }),
        });
      } catch (error) {
        reportActionError("Highlight item failed", error);
      }
    },
    [issueUiCommand, reportActionError],
  );

  const handleFocusMap = React.useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/focus-map", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Fit map failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleFocusRobot = React.useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/focus-robot", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Focus robot failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleClearFocus = React.useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/clear-focus", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Clear focus failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleToggleTeleop = React.useCallback(() => {
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

  const handleStopMotion = React.useCallback(async () => {
    try {
      await fetchJson("/api/teleop/stop", { method: "POST" });
    } catch {
      // Best-effort stop path. If the service is unavailable, there is nothing
      // useful to surface here during cleanup.
    }
  }, []);

  const handleStopDimos = React.useCallback(async () => {
    const confirmed = window.confirm("Stop the active DimOS run?");
    if (!confirmed) {
      return;
    }

    appendActivity("operator", "Stop DimOS", "Graceful shutdown requested.", "danger");
    setBusyAction("system-stop");
    setTeleopEnabled(false);
    try {
      await fetchJson("/api/system/stop", { method: "POST" });
      appendActivity("system", "DimOS stopping", "Stop signal sent to the active run.", "danger");
    } catch (error) {
      reportActionError("Stop DimOS failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleSubmitChatMessage = React.useCallback(
    async (message: string) => {
      appendActivity("operator", "Orchestrator prompt", message, "accent");
      try {
        const payload = await fetchJson<ChatState>("/api/chat", {
          method: "POST",
          body: JSON.stringify({ message }),
        });
        startTransition(() => {
          setState((previous) => ({ ...previous, chat: payload }));
        });
      } catch (error) {
        reportActionError("Chat request failed", error);
      }
    },
    [appendActivity, reportActionError],
  );

  const handleResetChat = React.useCallback(async () => {
    try {
      const payload = await fetchJson<ChatState>("/api/chat/reset", {
        method: "POST",
      });
      startTransition(() => {
        setState((previous) => ({ ...previous, chat: payload }));
      });
      appendActivity("system", "Orchestrator reset", "Chat history cleared.", "neutral");
    } catch (error) {
      reportActionError("Chat reset failed", error);
    }
  }, [appendActivity, reportActionError]);

  React.useEffect(() => {
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
        const message = error instanceof Error ? error.message : "Teleop request failed";
        if (teleopErrorMessageRef.current !== message) {
          teleopErrorMessageRef.current = message;
          reportActionError("Teleop command failed", error);
          appendActivity("system", "Teleop disabled", "Control path lost.", "danger");
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

  const poseLabel = formatPoseLabel(state);
  const povLabel = state.pov.updated_at ? formatTimestamp(state.pov.updated_at) : "No frame";
  const mapLabel = state.map ? formatTimestamp(state.map.updated_at) : "No map";
  return (
    <div className="app-shell">
      <div className="app-shell-noise" />

      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark">S</div>
          <div className="topbar-brand-copy">
            <h1>Navigator</h1>
            <p>Semantic map ops</p>
          </div>
        </div>

        <div className="topbar-status">
          {teleopEnabled ? <span className="toolbar-chip tone-danger">Teleop armed</span> : null}
          {state.inspection.status === "running" ? (
            <span className="toolbar-chip tone-running">Inspecting</span>
          ) : null}
          {state.chat.running ? (
            <span className="toolbar-chip tone-accent">Orchestrator thinking</span>
          ) : null}
          {!state.openai_configured ? (
            <span
              className="toolbar-chip tone-accent"
              title="OPENAI_API_KEY unset: Inspect saves the camera view; agent chat replies with a setup note. Add a key for full AI chat and VLM inspect."
            >
              OpenAI off
            </span>
          ) : null}
          {!state.yolo_runtime.inference_enabled ? (
            <span className="toolbar-chip tone-danger">YOLO off</span>
          ) : null}
          {state.yolo_runtime.mode !== "live" ? (
            <span className="toolbar-chip tone-accent">YOLO paused</span>
          ) : null}
        </div>

        <div className="topbar-actions">
          <div className="layout-toggle" aria-label="Dashboard layout">
            <button
              className={layoutMode === "duo" ? "is-active" : ""}
              onClick={() => {
                if (layoutMode !== "duo") {
                  setLayoutMode("duo");
                }
              }}
              type="button"
            >
              Duo
            </button>
            <button
              className={layoutMode === "trio" ? "is-active" : ""}
              onClick={() => {
                if (layoutMode !== "trio") {
                  setLayoutMode("trio");
                }
              }}
              type="button"
            >
              Trio
            </button>
          </div>

          <button
            className="action-button"
            disabled={busyAction !== null || state.inspection.status === "running"}
            onClick={() => {
              void handleInspectNow();
            }}
            title={
              state.openai_configured
                ? undefined
                : "Saves current view as a POI without VLM. Set OPENAI_API_KEY for AI inspect."
            }
            type="button"
          >
            {state.inspection.status === "running" ? "Inspecting" : "Inspect"}
          </button>

          <button
            aria-label="Save map"
            className="action-button secondary settings-cog-button"
            disabled={busyAction !== null || state.map === null}
            onClick={() => {
              void handleSaveMap();
            }}
            title="Save map"
            type="button"
          >
            <SaveMapGlyphs />
          </button>

          <button
            className={`action-button ${teleopEnabled ? "danger" : "success"}`}
            disabled={busyAction === "system-stop"}
            onClick={() => {
              handleToggleTeleop();
            }}
            type="button"
          >
            {teleopEnabled ? "Teleop On" : "Teleop Off"}
          </button>

          <button
            className="action-button danger"
            disabled={busyAction === "system-stop"}
            onClick={() => {
              void handleStopDimos();
            }}
            type="button"
          >
            {busyAction === "system-stop" ? "Stopping" : "Stop"}
          </button>

          <div className="topbar-menu" ref={controlsMenuRef}>
            <button
              aria-expanded={controlsMenuOpen}
              aria-haspopup="menu"
              aria-label="Settings"
              className="action-button secondary settings-cog-button"
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
                      void handleInspectionModeChange(event.target.value as ManualInspectionMode);
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
                    void handleYoloModeChange(state.yolo_runtime.mode === "live" ? "paused" : "live");
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
                    void handleYoloInferenceEnabledChange(!state.yolo_runtime.inference_enabled);
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
                    void handleLayerToggle("show_yolo", !state.layers.show_yolo);
                    setControlsMenuOpen(false);
                  }}
                  type="button"
                >
                  {state.layers.show_yolo ? "Hide YOLO layer" : "Show YOLO layer"}
                </button>
                <button
                  className="menu-item"
                  onClick={() => {
                    void handleLayerToggle("show_pois", !state.layers.show_pois);
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
                    busyAction !== null || (state.pois.length === 0 && state.yolo_objects.length === 0)
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
        </div>
      </header>

      <main className={`workspace workspace--${layoutMode}`}>
        <LiveFeedPanel
          connected={state.connected}
          frameLabel={povLabel}
          poseLabel={poseLabel}
          pov={state.pov}
        />

        <PanelShell
          aside={
            <div className="panel-chip-row">
              <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs font-medium leading-none text-slate-400">
                Updated {mapLabel}
              </span>
            </div>
          }
          bodyClassName="panel-body-stage"
          className="map-panel"
          kicker="Spatial"
          title="Map"
        >
          <MapPane
            layers={state.layers}
            map={state.map}
            robotOperatorHoverCard={defaultRobotOperatorHoverCard("slamass")}
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

        {layoutMode === "trio" ? (
          <OperatorRail
            activityEntries={activityEntries}
            busyAction={busyAction}
            chat={state.chat}
            items={semanticItems}
            onResetChat={() => {
              void handleResetChat();
            }}
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
            onSelectItem={(item) => {
              void handleSelectItem(item);
            }}
            selectedItem={state.ui.selected_item}
            selectedPreview={selectedPreview}
            onSubmitChatMessage={(message) => {
              void handleSubmitChatMessage(message);
            }}
          />
        ) : null}
      </main>

      {selectedPoi || selectedYoloObject ? (
        <div
          className="poi-modal-backdrop"
          onClick={() => {
            void handleSelectItem(null);
          }}
        >
          <div className="poi-modal" onClick={(event) => event.stopPropagation()}>
            {selectedPoi ? (
              <>
                <div className="poi-modal-media">
                  <img alt={selectedPoi.title} src={selectedPoi.hero_image_url} />
                </div>
                <div className="poi-modal-body">
                  <div className="poi-modal-header">
                    <div>
                      <p className="eyebrow">{selectedPoi.category}</p>
                      <h3>{selectedPoi.title}</h3>
                    </div>
                    <div className="poi-modal-header-actions">
                      <span className="score-pill">{selectedPoi.interest_score.toFixed(2)}</span>
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
                      Target {selectedPoi.target_x.toFixed(2)}, {selectedPoi.target_y.toFixed(2)}
                    </span>
                    <span>
                      View {selectedPoi.anchor_x.toFixed(2)}, {selectedPoi.anchor_y.toFixed(2)} |{" "}
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
                        void handleHighlightItem({ kind: "vlm_poi", entity_id: selectedPoi.poi_id });
                      }}
                      type="button"
                    >
                      Highlight
                    </button>
                    <button
                      className="action-button secondary"
                      onClick={() => {
                        void handleFocusItem({ kind: "vlm_poi", entity_id: selectedPoi.poi_id });
                      }}
                      type="button"
                    >
                      Focus
                    </button>
                    <button
                      className="action-button"
                      disabled={busyAction === `go-vlm_poi-${selectedPoi.poi_id}`}
                      onClick={() => {
                        void handleGoToItem({ kind: "vlm_poi", entity_id: selectedPoi.poi_id });
                      }}
                      type="button"
                    >
                      Go To
                    </button>
                    <button
                      className="action-button danger"
                      disabled={busyAction === `delete-vlm_poi-${selectedPoi.poi_id}`}
                      onClick={() => {
                        void handleDeleteItem({ kind: "vlm_poi", entity_id: selectedPoi.poi_id });
                      }}
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </>
            ) : null}

            {selectedYoloObject ? (
              <>
                <div className="poi-modal-media">
                  <img alt={selectedYoloObject.label} src={selectedYoloObject.hero_image_url} />
                </div>
                <div className="poi-modal-body">
                  <div className="poi-modal-header">
                    <div>
                      <p className="eyebrow">YOLO object</p>
                      <h3>{selectedYoloObject.label}</h3>
                    </div>
                    <div className="poi-modal-header-actions">
                      <span className="score-pill">
                        {(selectedYoloObject.best_confidence * 100).toFixed(0)}%
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
                    Stable YOLO object promoted from {selectedYoloObject.detections_count} detections.
                    Navigation will restore the best recorded viewing pose.
                  </p>

                  <div className="poi-meta">
                    <span>
                      Object {selectedYoloObject.world_x.toFixed(2)}, {selectedYoloObject.world_y.toFixed(2)}
                    </span>
                    <span>
                      View {selectedYoloObject.best_view_x.toFixed(2)}, {selectedYoloObject.best_view_y.toFixed(2)} |{" "}
                      {formatYaw(selectedYoloObject.best_view_yaw)}
                    </span>
                    <span>{formatTimestamp(selectedYoloObject.last_seen_at)}</span>
                  </div>

                  <div className="poi-tags">
                    <span>{selectedYoloObject.label}</span>
                    <span>{selectedYoloObject.detections_count} hits</span>
                    <span>{selectedYoloObject.size_x.toFixed(2)}m × {selectedYoloObject.size_y.toFixed(2)}m</span>
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
                      disabled={busyAction === `go-yolo_object-${selectedYoloObject.object_id}`}
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
                      disabled={busyAction === `delete-yolo_object-${selectedYoloObject.object_id}`}
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
