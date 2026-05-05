import React, {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { apiUrl, normalizeAppStateForDev } from "../apiBase";
import { isPolarisDemoCameraCaptureRef, mergePoi, mergeYoloObject } from "../semanticItems";
import type {
  AppState,
  ChatState,
  InspectionSettings,
  InspectionState,
  ManualInspectionMode,
  Poi,
  SemanticItemRef,
  UiCameraState,
  UiState,
  YoloObject,
} from "../types";
import { applyFitOverviewCameraToAppState, DEFAULT_MAP_ZOOM } from "../mapViewport";
import { mergeUiPreferringLocalCamera } from "../uiStateMerge";
import { fetchJson } from "./fetchJson";

const LAYOUT_STORAGE_KEY = "slamass-layout-mode";
const MAX_ACTIVITY_ENTRIES = 10;

export type LayoutMode = "duo" | "trio";

type ActivityRole = "operator" | "system";
type ActivityTone = "neutral" | "accent" | "success" | "danger";

export type ActivityEntry = {
  id: string;
  role: ActivityRole;
  tone: ActivityTone;
  title: string;
  detail: string;
  timestamp: string;
};

function formatClock(date: Date): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
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

function applyUiState(previous: UiState, next: UiState): UiState {
  return next.revision >= previous.revision ? next : previous;
}

export type SlamassApiStatus = "loading" | "ok" | "error";

export function useNavigatorSlamassState(): {
  state: AppState;
  /** Whether /api/state has been reached; "error" if the first load failed (CORS, wrong port, sidecar down). */
  slamassApiStatus: SlamassApiStatus;
  layoutMode: LayoutMode;
  setLayoutMode: (mode: LayoutMode) => void;
  busyAction: string | null;
  activityEntries: ActivityEntry[];
  queueCameraSync: (camera: UiCameraState) => void;
  handleSelectItem: (item: SemanticItemRef | null) => void;
  handleFocusItem: (item: SemanticItemRef) => void;
  handleFocusMap: () => void;
  handleFocusRobot: () => void;
  handleClearFocus: () => void;
  handleNavigate: (x: number, y: number) => void;
  handleHighlightItem: (item: SemanticItemRef) => void;
  handleGoToItem: (item: SemanticItemRef) => void;
  handleDeleteItem: (item: SemanticItemRef) => void;
  handleSubmitChatMessage: (message: string) => void;
  handleResetChat: () => void;
  handleInspectNow: () => void;
  handleSaveMap: () => void;
  handleInspectionModeChange: (mode: ManualInspectionMode) => void;
  handleYoloModeChange: (mode: "live" | "paused") => void;
  handleYoloInferenceEnabledChange: (enabled: boolean) => void;
  handleLayerToggle: (field: "show_pois" | "show_yolo", value: boolean) => void;
  handleClearLowLevelMapMemory: () => void;
  handleClearSemanticMemory: () => void;
  handleStopDimos: () => void;
  handleStopMotion: () => void;
  appendActivity: (
    role: ActivityRole,
    title: string,
    detail: string,
    tone?: ActivityTone,
  ) => void;
  reportActionError: (title: string, error: unknown) => void;
} {
  const [state, setState] = useState<AppState>(emptyState);
  const [slamassApiStatus, setSlamassApiStatus] =
    useState<SlamassApiStatus>("loading");
  const [layoutMode, setLayoutModeState] =
    useState<LayoutMode>(getInitialLayoutMode);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [activityEntries, setActivityEntries] = useState<ActivityEntry[]>(
    () => [
      {
        id: "boot",
        role: "system",
        tone: "neutral",
        title: "Navigator",
        detail: "Same workspace layout as the main Navigator dashboard.",
        timestamp: formatClock(new Date()),
      },
    ],
  );
  const activityCounterRef = useRef(1);
  const cameraSyncTimerRef = useRef<number | null>(null);
  /** Latest camera from wheel/pan while debounced PUT is pending or in flight (SSE must not stomp it). */
  const pendingLocalCameraRef = useRef<UiCameraState | null>(null);
  const cameraPutInFlightRef = useRef(false);
  const sessionMapFitDoneRef = useRef(false);

  const mergeUiState = useCallback((nextUi: UiState) => {
    startTransition(() => {
      setState((previous) => ({
        ...previous,
        ui: applyUiState(previous.ui, nextUi),
      }));
    });
  }, []);

  const issueUiCommand = useCallback(
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

  const setLayoutMode = useCallback((next: LayoutMode) => {
    setLayoutModeState(next);
    persistLayoutMode(next);
  }, []);

  const appendActivity = useCallback(
    (
      role: ActivityRole,
      title: string,
      detail: string,
      tone: ActivityTone = "neutral",
    ) => {
      const entry: ActivityEntry = {
        id: `activity-${activityCounterRef.current}`,
        role,
        tone,
        title,
        detail,
        timestamp: formatClock(new Date()),
      };
      activityCounterRef.current += 1;
      setActivityEntries((previous) =>
        [...previous, entry].slice(-MAX_ACTIVITY_ENTRIES),
      );
    },
    [],
  );

  const reportActionError = useCallback(
    (title: string, error: unknown) => {
      const message = error instanceof Error ? error.message : "Request failed";
      appendActivity("system", title, message, "danger");
    },
    [appendActivity],
  );

  const queueCameraSync = useCallback(
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

  useEffect(() => {
    let cancelled = false;

    const loadStateOnce = async (): Promise<boolean> => {
      try {
        const data = await fetchJson<AppState>("/api/state");
        if (!cancelled) {
          setSlamassApiStatus("ok");
          const normalized = normalizeAppStateForDev(data);
          const withFit = data.map
            ? applyFitOverviewCameraToAppState(normalized)
            : normalized;
          let postFocusMapFromLoad = false;
          if (data.map && !sessionMapFitDoneRef.current) {
            sessionMapFitDoneRef.current = true;
            postFocusMapFromLoad = true;
          }
          startTransition(() => {
            setState(withFit);
          });
          if (postFocusMapFromLoad) {
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
        }
        return true;
      } catch {
        if (!cancelled) {
          setSlamassApiStatus("error");
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

    const onStateUpdated = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as Partial<AppState>;
      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            ...payload,
            pov: payload.pov
              ? { ...previous.pov, ...payload.pov }
              : previous.pov,
            yolo_runtime: payload.yolo_runtime ?? previous.yolo_runtime,
            layers: payload.layers ?? previous.layers,
            inspection_settings:
              payload.inspection_settings ?? previous.inspection_settings,
          }),
        );
      });
    };

    const onMapUpdated = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as AppState["map"];
      if (payload === null) {
        sessionMapFitDoneRef.current = false;
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
      if (shouldSessionFit) {
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
    };

    const onPoiUpserted = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as Poi;
      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            pois: mergePoi(previous.pois, payload),
          }),
        );
      });
    };

    const onPoiDeleted = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as { poi_id: string };
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          pois: previous.pois.filter((poi) => poi.poi_id !== payload.poi_id),
        }));
      });
    };

    const onYoloUpserted = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as YoloObject;
      startTransition(() => {
        setState((previous) =>
          normalizeAppStateForDev({
            ...previous,
            yolo_objects: mergeYoloObject(previous.yolo_objects, payload),
          }),
        );
      });
    };

    const onYoloDeleted = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as { object_id: string };
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          yolo_objects: previous.yolo_objects.filter(
            (object) => object.object_id !== payload.object_id,
          ),
        }));
      });
    };

    const onInspectionUpdated = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as InspectionState;
      startTransition(() => {
        setState((previous) => ({ ...previous, inspection: payload }));
      });
    };

    const onChatUpdated = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as ChatState;
      startTransition(() => {
        setState((previous) => ({ ...previous, chat: payload }));
      });
    };

    const onUiUpdated = (event: MessageEvent): void => {
      const payload = JSON.parse(event.data as string) as UiState;
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
    };

    source.addEventListener("state_updated", onStateUpdated);
    source.addEventListener("map_updated", onMapUpdated);
    source.addEventListener("poi_upserted", onPoiUpserted);
    source.addEventListener("poi_deleted", onPoiDeleted);
    source.addEventListener("yolo_object_upserted", onYoloUpserted);
    source.addEventListener("yolo_object_deleted", onYoloDeleted);
    source.addEventListener("inspection_updated", onInspectionUpdated);
    source.addEventListener("chat_state_updated", onChatUpdated);
    source.addEventListener("ui_state_updated", onUiUpdated);

    return () => {
      cancelled = true;
      source.removeEventListener("state_updated", onStateUpdated);
      source.removeEventListener("map_updated", onMapUpdated);
      source.removeEventListener("poi_upserted", onPoiUpserted);
      source.removeEventListener("poi_deleted", onPoiDeleted);
      source.removeEventListener("yolo_object_upserted", onYoloUpserted);
      source.removeEventListener("yolo_object_deleted", onYoloDeleted);
      source.removeEventListener("inspection_updated", onInspectionUpdated);
      source.removeEventListener("chat_state_updated", onChatUpdated);
      source.removeEventListener("ui_state_updated", onUiUpdated);
      source.close();
      if (cameraSyncTimerRef.current !== null) {
        window.clearTimeout(cameraSyncTimerRef.current);
      }
    };
  }, [mergeUiState]);

  const handleSelectItem = useCallback(
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

  const handleFocusItem = useCallback(
    async (item: SemanticItemRef) => {
      if (isPolarisDemoCameraCaptureRef(item)) {
        return;
      }
      try {
        await issueUiCommand(
          `/api/ui/focus-item/${item.kind}/${item.entity_id}`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
      } catch (error) {
        reportActionError("Focus item failed", error);
      }
    },
    [issueUiCommand, reportActionError],
  );

  const handleFocusMap = useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/focus-map", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Fit map failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleFocusRobot = useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/focus-robot", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Focus robot failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleClearFocus = useCallback(async () => {
    try {
      await issueUiCommand("/api/ui/clear-focus", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch (error) {
      reportActionError("Clear focus failed", error);
    }
  }, [issueUiCommand, reportActionError]);

  const handleNavigate = useCallback(
    async (x: number, y: number) => {
      appendActivity(
        "operator",
        "Navigate",
        `${x.toFixed(2)}, ${y.toFixed(2)}`,
        "accent",
      );
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

  const handleGoToItem = useCallback(
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
      appendActivity(
        "operator",
        "Go to item",
        item.kind === "vlm_poi" ? "POI" : "YOLO object",
        "accent",
      );
      setBusyAction(actionKey);
      try {
        if (item.kind === "vlm_poi") {
          await fetchJson(`/api/pois/${item.entity_id}/go`, { method: "POST" });
        } else {
          await fetchJson(`/api/yolo-objects/${item.entity_id}/go`, {
            method: "POST",
          });
        }
      } catch (error) {
        reportActionError("Go To request failed", error);
      } finally {
        setBusyAction(null);
      }
    },
    [appendActivity, reportActionError],
  );

  const handleDeleteItem = useCallback(
    async (item: SemanticItemRef) => {
      const confirmed = window.confirm(
        "Delete this semantic item from the Navigator map?",
      );
      if (!confirmed) {
        return;
      }
      const actionKey = `delete-${item.kind}-${item.entity_id}`;
      setBusyAction(actionKey);
      try {
        if (item.kind === "vlm_poi") {
          await fetchJson(`/api/pois/${item.entity_id}/delete`, {
            method: "POST",
          });
        } else {
          await fetchJson(`/api/yolo-objects/${item.entity_id}/delete`, {
            method: "POST",
          });
        }
      } catch (error) {
        reportActionError("Delete item failed", error);
      } finally {
        setBusyAction(null);
      }
    },
    [reportActionError],
  );

  const handleHighlightItem = useCallback(
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

  const handleSubmitChatMessage = useCallback(
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

  const handleResetChat = useCallback(async () => {
    try {
      const payload = await fetchJson<ChatState>("/api/chat/reset", {
        method: "POST",
      });
      startTransition(() => {
        setState((previous) => ({ ...previous, chat: payload }));
      });
      appendActivity(
        "system",
        "Orchestrator reset",
        "Chat history cleared.",
        "neutral",
      );
    } catch (error) {
      reportActionError("Chat reset failed", error);
    }
  }, [appendActivity, reportActionError]);

  const handleInspectNow = useCallback(async () => {
    appendActivity(
      "operator",
      "Inspect",
      "Manual semantic inspection started.",
      "accent",
    );
    setBusyAction("inspect");
    try {
      await fetchJson("/api/inspect/now", { method: "POST" });
    } catch (error) {
      reportActionError("Inspect request failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleSaveMap = useCallback(async () => {
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

  const handleInspectionModeChange = useCallback(
    async (mode: ManualInspectionMode) => {
      try {
        const nextSettings = await fetchJson<InspectionSettings>(
          "/api/inspection-settings",
          {
            method: "PUT",
            body: JSON.stringify({ manual_mode: mode }),
          },
        );
        startTransition(() => {
          setState((previous) => ({
            ...previous,
            inspection_settings: nextSettings,
          }));
        });
      } catch (error) {
        reportActionError("Inspect mode update failed", error);
      }
    },
    [reportActionError],
  );

  const handleYoloModeChange = useCallback(
    async (mode: "live" | "paused") => {
      try {
        const payload = await fetchJson<AppState["yolo_runtime"]>(
          "/api/yolo/runtime",
          {
            method: "PUT",
            body: JSON.stringify({ mode }),
          },
        );
        startTransition(() => {
          setState((previous) => ({ ...previous, yolo_runtime: payload }));
        });
      } catch (error) {
        reportActionError("YOLO mode update failed", error);
      }
    },
    [reportActionError],
  );

  const handleYoloInferenceEnabledChange = useCallback(
    async (inferenceEnabled: boolean) => {
      try {
        const payload = await fetchJson<AppState["yolo_runtime"]>(
          "/api/yolo/runtime",
          {
            method: "PUT",
            body: JSON.stringify({ inference_enabled: inferenceEnabled }),
          },
        );
        startTransition(() => {
          setState((previous) => ({ ...previous, yolo_runtime: payload }));
        });
      } catch (error) {
        reportActionError("YOLO inference update failed", error);
      }
    },
    [reportActionError],
  );

  const handleLayerToggle = useCallback(
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

  const handleClearLowLevelMapMemory = useCallback(async () => {
    const confirmed = window.confirm(
      "Clear the low-level persistent map memory?\n\nThis removes the active occupancy map only. Semantic POIs and YOLO objects are kept.",
    );
    if (!confirmed) {
      return;
    }

    appendActivity(
      "operator",
      "Clear map memory",
      "Low-level map reset requested.",
      "danger",
    );
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
      appendActivity(
        "system",
        "Map memory cleared",
        "Active occupancy map removed.",
        "danger",
      );
    } catch (error) {
      reportActionError("Clear map memory failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleClearSemanticMemory = useCallback(async () => {
    const confirmed = window.confirm(
      "Clear semantic memory?\n\nThis removes VLM POIs, YOLO objects, semantic observations, and chat history. The low-level occupancy map stays intact.",
    );
    if (!confirmed) {
      return;
    }

    appendActivity(
      "operator",
      "Clear semantic memory",
      "Full semantic reset requested.",
      "danger",
    );
    setBusyAction("clear-semantic-memory");
    try {
      await fetchJson("/api/memory/clear-semantic", { method: "POST" });
      startTransition(() => {
        setState((previous) => ({
          ...previous,
          pois: [],
          yolo_objects: [],
          chat: { running: false, messages: [] },
        }));
      });
      appendActivity(
        "system",
        "Semantic memory cleared",
        "POIs and YOLO objects removed.",
        "danger",
      );
    } catch (error) {
      reportActionError("Clear semantic memory failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleStopDimos = useCallback(async () => {
    const confirmed = window.confirm("Stop the active DimOS run?");
    if (!confirmed) {
      return;
    }

    appendActivity(
      "operator",
      "Stop DimOS",
      "Graceful shutdown requested.",
      "danger",
    );
    setBusyAction("system-stop");
    try {
      await fetchJson("/api/system/stop", { method: "POST" });
      appendActivity(
        "system",
        "DimOS stopping",
        "Stop signal sent to the active run.",
        "danger",
      );
    } catch (error) {
      reportActionError("Stop DimOS failed", error);
    } finally {
      setBusyAction(null);
    }
  }, [appendActivity, reportActionError]);

  const handleStopMotion = useCallback(async () => {
    try {
      await fetchJson("/api/teleop/stop", { method: "POST" });
    } catch {
      /* best-effort */
    }
  }, []);

  return {
    state,
    slamassApiStatus,
    layoutMode,
    setLayoutMode,
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
  };
}
