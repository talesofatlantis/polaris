import React from "react";
import {
  ArrowsPointingInIcon,
  MinusIcon,
  PlusIcon,
} from "@heroicons/react/24/outline";

import {
  buildViewport,
  clampZoom,
  fitOverviewCamera,
  MAX_MAP_ZOOM,
  MIN_MAP_ZOOM,
  panCamera,
  screenToWorld,
  worldPositionForAdditionalOperator,
  worldToImagePixels,
  zoomCameraAtScreenPoint,
  type BuildViewportOptions,
  type MapViewport,
} from "./mapViewport";
import { refFromPoi, refFromYoloObject, semanticKey } from "./semanticItems";
import type { RobotOperatorHoverCard } from "./robotOperatorLabel";
import {
  LayerVisibility,
  MapState,
  Poi,
  RobotPose,
  SemanticItemRef,
  UiCameraState,
  UiState,
  YoloObject,
} from "./types";

function useSize<T extends HTMLElement>(
  /** Re-run layout measure when this identity changes (e.g. map becomes available). */
  measureKey?: unknown,
): [React.RefObject<T>, { width: number; height: number }] {
  const ref = React.useRef<T>(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });

  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) {
      return undefined;
    }
    const apply = (width: number, height: number) => {
      setSize((previous) =>
        previous.width === width && previous.height === height ? previous : { width, height },
      );
    };
    const measure = () => {
      apply(el.clientWidth, el.clientHeight);
    };
    measure();
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const target = entry.target as HTMLElement;
      apply(target.clientWidth, target.clientHeight);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [measureKey]);

  return [ref, size];
}

function isInsideMapFrame(
  screenX: number,
  screenY: number,
  imageLeft: number,
  imageTop: number,
  imageWidth: number,
  imageHeight: number,
): boolean {
  return (
    screenX >= imageLeft &&
    screenX <= imageLeft + imageWidth &&
    screenY >= imageTop &&
    screenY <= imageTop + imageHeight
  );
}

export type MapAdditionalRobotMarker = {
  id: string;
  card: RobotOperatorHoverCard;
  /** One-shot emphasis (e.g. right after create flow). */
  deployPop?: boolean;
};

function RobotOperatorMarkerView(props: {
  map: MapState;
  viewport: MapViewport;
  worldX: number;
  worldY: number;
  card: RobotOperatorHoverCard;
  operatorFleetGo2Hover?: boolean;
  deployPop?: boolean;
  gamecardDomId: string;
  /** 0 = primary live robot; higher draws above siblings. */
  stackIndex: number;
  onPointerDownStop: (event: React.SyntheticEvent) => void;
}): React.ReactElement {
  const {
    map,
    viewport,
    worldX,
    worldY,
    card,
    operatorFleetGo2Hover = false,
    deployPop = false,
    gamecardDomId,
    stackIndex,
    onPointerDownStop,
  } = props;
  const [robotX, robotY] = worldToImagePixels(map, viewport, worldX, worldY);
  const statusLabel =
    card.active === "blue"
      ? "Standby"
      : card.active === "grey"
        ? "Inactive"
        : card.active === "green"
          ? "Active"
          : undefined;
  return (
    <div
      aria-describedby={gamecardDomId}
      aria-label={`Robot — ${card.instanceName}`}
      className={[
        "robot-marker",
        operatorFleetGo2Hover ? "robot-marker--fleet-hover" : "",
        deployPop ? "robot-marker--deploy-pop" : "",
        stackIndex > 0 ? "robot-marker--additional" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onPointerDown={onPointerDownStop}
      style={{
        left: `${robotX}px`,
        top: `${robotY}px`,
        transform: "translate(-50%, -50%)",
        zIndex: 3 + stackIndex,
      }}
      tabIndex={0}
    >
      <span className="robot-marker-pulse-ring robot-marker-pulse-ring--1" />
      <span className="robot-marker-pulse-ring robot-marker-pulse-ring--2" />
      <div className="robot-marker-disc" />
      <div className="robot-marker-gamecard" id={gamecardDomId} role="tooltip">
        <div className="robot-marker-gamecard-media">
          <img
            alt={card.imageAlt}
            className="robot-marker-gamecard-img"
            decoding="async"
            draggable={false}
            src={card.imageUrl}
          />
        </div>
        <div className="robot-marker-gamecard-body">
          <div className="robot-marker-gamecard-title-row">
            <span className="robot-marker-gamecard-title">{card.instanceName}</span>
            {card.active ? (
              <span
                aria-label={statusLabel}
                className="robot-marker-gamecard-status"
                role="status"
              >
                <span
                  aria-hidden
                  className={`robot-marker-gamecard-dot robot-marker-gamecard-dot--${card.active}`}
                />
              </span>
            ) : null}
          </div>
          {card.modelTitle ? (
            <p className="robot-marker-gamecard-model">{card.modelTitle}</p>
          ) : null}
          {card.typeLine ? (
            <p className="robot-marker-gamecard-meta">
              <span className="robot-marker-gamecard-meta-label">Type</span>{" "}
              <span className="robot-marker-gamecard-meta-value">{card.typeLine}</span>
            </p>
          ) : null}
          {card.location ? (
            <p className="robot-marker-gamecard-meta">
              <span className="robot-marker-gamecard-meta-label">Location</span>{" "}
              <span className="robot-marker-gamecard-meta-value">{card.location}</span>
            </p>
          ) : null}
          {card.task ? (
            <p className="robot-marker-gamecard-meta">
              <span className="robot-marker-gamecard-meta-label">Task</span>{" "}
              <span className="robot-marker-gamecard-meta-value">{card.task}</span>
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

type MapPaneProps = {
  map: MapState | null;
  robotPose: RobotPose | null;
  /** Compact operator-style hover card on the robot marker. */
  robotOperatorHoverCard: RobotOperatorHoverCard;
  path: Array<[number, number]>;
  pois: Poi[];
  yoloObjects: YoloObject[];
  layers: LayerVisibility;
  ui: UiState;
  onCameraChange: (camera: UiCameraState) => void;
  onNavigate: (x: number, y: number) => void;
  onSelectItem: (item: SemanticItemRef | null) => void;
  onFocusItem: (item: SemanticItemRef) => void;
  onFocusMap: () => void;
  onFocusRobot: () => void;
  onClearFocus: () => void;
  /** When false, hide Fit / Robot / Selected / Clear. Default true. */
  showMapToolbar?: boolean;
  /**
   * Re-apply fit overview once this pane gets a non-zero layout size (navigator split layout).
   * Centers the map in the map column after flex/grid height is known, independent of the sidebar.
   */
  refitOnLayoutReady?: boolean;
  /**
   * Navigator: extra controls pinned top-right inside the map surface; map widgets move to bottom-right.
   */
  mapOverlayTopRight?: React.ReactNode;
  /**
   * Navigator: keep zoom/fit widgets on the bottom-right without a map-surface top overlay
   * (e.g. when actions live in `PanelShell` header `aside`).
   */
  pinViewModeControlsBottom?: boolean;
  /**
   * Where the camera center is pinned on the map surface (0–1). Default 0.5 / 0.5 (geometric center).
   * Navigator uses a lower `y` so the map sits higher under the panel title on load.
   */
  viewportScreenAnchorX?: number;
  viewportScreenAnchorY?: number;
  /**
   * When true, show the robot operator gamecard as if the map marker were hovered (e.g. first Go2 row
   * hover in the navigator Operators list).
   */
  operatorFleetGo2Hover?: boolean;
  /**
   * Extra operators on the map (e.g. newly deployed from create flow), placed near the live robot or
   * map center.
   */
  additionalRobotMarkers?: MapAdditionalRobotMarker[];
};

type DragState = {
  pointerId: number;
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  moved: boolean;
  startCamera: UiCameraState;
};

export function MapPane(props: MapPaneProps): React.ReactElement {
  const {
    map,
    robotPose,
    robotOperatorHoverCard,
    path,
    pois,
    yoloObjects,
    layers,
    ui,
    onCameraChange,
    onNavigate,
    onSelectItem,
    onFocusItem,
    onFocusMap,
    onFocusRobot,
    onClearFocus,
    showMapToolbar = true,
    refitOnLayoutReady = false,
    mapOverlayTopRight,
    pinViewModeControlsBottom = false,
    viewportScreenAnchorX,
    viewportScreenAnchorY,
    operatorFleetGo2Hover = false,
    additionalRobotMarkers,
  } = props;

  const viewportBuildOptions = React.useMemo((): BuildViewportOptions => {
    return {
      screenAnchorX: viewportScreenAnchorX ?? 0.5,
      screenAnchorY: viewportScreenAnchorY ?? 0.5,
    };
  }, [viewportScreenAnchorX, viewportScreenAnchorY]);

  const mapMeasureKey = map ? `${map.width}x${map.height}` : null;
  const [containerRef, size] = useSize<HTMLDivElement>(mapMeasureKey);
  const dragStateRef = React.useRef<DragState | null>(null);
  const cameraRef = React.useRef(ui.camera);

  React.useEffect(() => {
    cameraRef.current = ui.camera;
  }, [ui.camera]);

  const layoutRefitKeyRef = React.useRef<string>("");
  React.useLayoutEffect(() => {
    if (!map) {
      layoutRefitKeyRef.current = "";
      return;
    }
    if (!refitOnLayoutReady) {
      return;
    }
    if (size.width <= 0 || size.height <= 0) {
      return;
    }
    const key = `${map.width}x${map.height}`;
    if (layoutRefitKeyRef.current === key) {
      return;
    }
    layoutRefitKeyRef.current = key;
    onCameraChange(fitOverviewCamera(map));
  }, [map, onCameraChange, refitOnLayoutReady, size.height, size.width]);

  React.useEffect(() => {
    const el = containerRef.current;
    if (!el || !map) {
      return undefined;
    }

    const onWheel = (event: WheelEvent): void => {
      const rect = el.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      if (width <= 0 || height <= 0) {
        return;
      }
      event.preventDefault();
      const cam = cameraRef.current;
      const viewportNow = buildViewport(map, width, height, cam, viewportBuildOptions);
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
      const nextZoom = clampZoom(cam.zoom * factor);
      if (Math.abs(nextZoom - cam.zoom) < 1e-6) {
        return;
      }
      onCameraChange(
        zoomCameraAtScreenPoint(map, cam, localX, localY, nextZoom, viewportNow),
      );
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
    };
  }, [map, onCameraChange, viewportBuildOptions]);

  const viewport = React.useMemo(() => {
    if (!map || size.width <= 0 || size.height <= 0) {
      return null;
    }
    return buildViewport(map, size.width, size.height, ui.camera, viewportBuildOptions);
  }, [map, size.height, size.width, ui.camera, viewportBuildOptions]);

  /** Identity of the map *grid* (not preview bumps). Live costmap updates change `image_version` often;
   * including that in the fade key caused the raster to opacity-flash on every SLAM frame. */
  const mapStructureKey = React.useMemo(() => {
    if (!map) {
      return null;
    }
    return `${map.map_id}|${map.width}|${map.height}|${map.resolution}|${map.origin_x}|${map.origin_y}`;
  }, [map?.map_id, map?.height, map?.origin_x, map?.origin_y, map?.resolution, map?.width]);
  const [mapLayerVisible, setMapLayerVisible] = React.useState(false);
  const mapImageRef = React.useRef<HTMLImageElement | null>(null);

  React.useEffect(() => {
    setMapLayerVisible(false);
  }, [mapStructureKey]);

  /** Cached images often skip `onLoad` after a Strict Mode remount — unstick the fade-in layer. */
  React.useLayoutEffect(() => {
    if (!mapStructureKey) {
      return;
    }
    const el = mapImageRef.current;
    if (el?.complete && el.naturalWidth > 0) {
      setMapLayerVisible(true);
    }
  }, [mapStructureKey]);

  /** Last resort if load events never fire (proxy hiccup, suspended tab, etc.). */
  React.useEffect(() => {
    if (!mapStructureKey) {
      return;
    }
    const id = window.setTimeout(() => {
      setMapLayerVisible(true);
    }, 3500);
    return () => window.clearTimeout(id);
  }, [mapStructureKey]);

  const mapStackFrameStyle = React.useMemo(() => {
    if (!viewport) {
      return undefined;
    }
    return {
      position: "absolute" as const,
      width: `${viewport.imageWidth}px`,
      height: `${viewport.imageHeight}px`,
      left: `${Math.round(viewport.imageLeft)}px`,
      top: `${Math.round(viewport.imageTop)}px`,
    };
  }, [viewport]);

  const selectedKey = semanticKey(ui.selected_item);

  const activePois = React.useMemo(
    () => pois.filter((poi) => poi.status !== "deleted"),
    [pois],
  );

  const activeYoloObjects = React.useMemo(
    () => yoloObjects.filter((object) => object.status !== "deleted"),
    [yoloObjects],
  );

  const stopEvent = React.useCallback((event: React.SyntheticEvent) => {
    event.stopPropagation();
  }, []);

  const additionalMarkerWorldPositions = React.useMemo(() => {
    if (!map || !additionalRobotMarkers?.length) {
      return [];
    }
    return additionalRobotMarkers.map((_, index) =>
      worldPositionForAdditionalOperator(map, robotPose, index),
    );
  }, [additionalRobotMarkers, map, robotPose]);

  const showFloatingZoomFit = Boolean(map && viewport);

  const atMinZoom = map ? ui.camera.zoom <= MIN_MAP_ZOOM + 1e-6 : false;
  const atMaxZoom = map ? ui.camera.zoom >= MAX_MAP_ZOOM - 1e-6 : false;

  const handleZoomIn = React.useCallback(() => {
    if (!map || !viewport) {
      return;
    }
    const cx = size.width * viewport.screenAnchorX;
    const cy = size.height * viewport.screenAnchorY;
    onCameraChange(
      zoomCameraAtScreenPoint(map, ui.camera, cx, cy, ui.camera.zoom * 1.1, viewport),
    );
  }, [map, onCameraChange, size.height, size.width, ui.camera, viewport]);

  const handleZoomOut = React.useCallback(() => {
    if (!map || !viewport) {
      return;
    }
    const cx = size.width * viewport.screenAnchorX;
    const cy = size.height * viewport.screenAnchorY;
    onCameraChange(
      zoomCameraAtScreenPoint(map, ui.camera, cx, cy, ui.camera.zoom / 1.1, viewport),
    );
  }, [map, onCameraChange, size.height, size.width, ui.camera, viewport]);

  const preventNativeDrag = React.useCallback((event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
  }, []);

  const handlePointerDown = React.useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!map || !viewport || event.button !== 0) {
        return;
      }
      const x = event.clientX;
      const y = event.clientY;
      dragStateRef.current = {
        pointerId: event.pointerId,
        startX: x,
        startY: y,
        lastX: x,
        lastY: y,
        moved: false,
        startCamera: ui.camera,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [map, ui.camera, viewport],
  );

  const handlePointerMove = React.useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!map || !viewport) {
        return;
      }
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }
      const ddx = event.clientX - dragState.lastX;
      const ddy = event.clientY - dragState.lastY;
      if (ddx === 0 && ddy === 0) {
        return;
      }
      dragState.lastX = event.clientX;
      dragState.lastY = event.clientY;
      dragState.moved = true;
      if (size.width <= 0 || size.height <= 0) {
        return;
      }
      const vp = buildViewport(map, size.width, size.height, cameraRef.current, viewportBuildOptions);
      const next = panCamera(map, cameraRef.current, ddx, ddy, vp);
      cameraRef.current = next;
      onCameraChange(next);
    },
    [map, onCameraChange, size.height, size.width, viewport, viewportBuildOptions],
  );

  const handlePointerUp = React.useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!map || !viewport) {
        return;
      }
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }
      dragStateRef.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);

      if (dragState.moved) {
        return;
      }

      const rect = event.currentTarget.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      if (
        !isInsideMapFrame(
          localX,
          localY,
          viewport.imageLeft,
          viewport.imageTop,
          viewport.imageWidth,
          viewport.imageHeight,
        )
      ) {
        return;
      }

      const [worldX, worldY] = screenToWorld(map, viewport, localX, localY);
      onSelectItem(null);
      onNavigate(worldX, worldY);
    },
    [map, onNavigate, onSelectItem, viewport],
  );

  const handlePointerCancel = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return;
    }
    dragStateRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const showMapTopOverlay = Boolean(mapOverlayTopRight);
  const useSplitChromeLayout = showMapTopOverlay || pinViewModeControlsBottom;

  const mapToolbarMainButtons = showMapToolbar ? (
    <>
      <button
        className="map-tool"
        disabled={!robotPose}
        onClick={onFocusRobot}
        type="button"
      >
        Robot
      </button>
      <button
        className="map-tool"
        disabled={!ui.selected_item}
        onClick={() => {
          if (ui.selected_item) {
            onFocusItem(ui.selected_item);
          }
        }}
        type="button"
      >
        Selected
      </button>
      <button
        className="map-tool"
        disabled={!ui.selected_item && ui.highlighted_items.length === 0}
        onClick={onClearFocus}
        type="button"
      >
        Clear
      </button>
    </>
  ) : null;

  const floatingZoomFitControls = showFloatingZoomFit ? (
    <div className="map-floating-controls">
      <div aria-label="Zoom map" className="map-zoom-pill" role="group">
        <button
          aria-label="Zoom in"
          className="map-zoom-pill-btn"
          disabled={atMaxZoom}
          onClick={handleZoomIn}
          type="button"
        >
          <PlusIcon aria-hidden className="map-zoom-pill-icon" />
        </button>
        <div aria-hidden className="map-zoom-pill-divider" />
        <button
          aria-label="Zoom out"
          className="map-zoom-pill-btn"
          disabled={atMinZoom}
          onClick={handleZoomOut}
          type="button"
        >
          <MinusIcon aria-hidden className="map-zoom-pill-icon" />
        </button>
      </div>
      <button
        aria-label="Fit map to view"
        className="map-fit-floating-btn"
        onClick={onFocusMap}
        title="Fit map"
        type="button"
      >
        <ArrowsPointingInIcon aria-hidden className="map-fit-floating-icon" />
      </button>
    </div>
  ) : null;

  const hasMapChrome =
    showMapToolbar || showMapTopOverlay || pinViewModeControlsBottom;

  const mapChromeBlocks =
    !hasMapChrome ? null : useSplitChromeLayout ? (
      <>
        {showMapTopOverlay ? (
          <div className="map-chrome map-chrome--overlay-top-right" onPointerDown={stopEvent}>
            <div className="map-overlay-top-actions">{mapOverlayTopRight}</div>
          </div>
        ) : null}
        {showMapToolbar || pinViewModeControlsBottom ? (
          <div className="map-chrome map-chrome--view-controls-bottom">
            <div
              className="map-toolbar map-toolbar--inset-bottom map-toolbar--with-floating-widgets"
              onPointerDown={stopEvent}
            >
              {mapToolbarMainButtons}
              {floatingZoomFitControls}
            </div>
          </div>
        ) : null}
      </>
    ) : (
      <>
        <div className="map-chrome">
          <div className="map-toolbar" onPointerDown={stopEvent}>
            {mapToolbarMainButtons}
          </div>
        </div>
        {floatingZoomFitControls ? (
          <div className="map-chrome map-chrome--floating-map-widgets" onPointerDown={stopEvent}>
            {floatingZoomFitControls}
          </div>
        ) : null}
      </>
    );

  return (
    <div
      className="map-surface"
      onDragStart={preventNativeDrag}
      onPointerCancel={handlePointerCancel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      ref={containerRef}
    >
      {mapChromeBlocks}
      {!map || !viewport ? (
        <div className="panel-empty panel-empty--map-loading">
          <h3>Navigator map not ready</h3>
          <p>Start the Go2 stack and wait for the service to ingest raw costmap updates.</p>
        </div>
      ) : (
        <>
          <div
            className={[
              "map-rendered-stack",
              mapLayerVisible ? "map-rendered-stack--visible" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            style={mapStackFrameStyle}
          >
            <img
              ref={mapImageRef}
              alt="Navigator occupancy map"
              className="map-image"
              draggable={false}
              onDragStart={preventNativeDrag}
              onError={() => {
                setMapLayerVisible(true);
              }}
              onLoad={() => {
                setMapLayerVisible(true);
              }}
              src={map.image_url}
            />
            <svg
              className="map-overlay map-overlay-anchored"
              preserveAspectRatio="none"
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: "100%",
                height: "100%",
              }}
              viewBox={`0 0 ${viewport.imageWidth} ${viewport.imageHeight}`}
            >
            {path.length > 1 && (
              <polyline
                className="path-line"
                points={path
                  .map(([x, y]) => worldToImagePixels(map, viewport, x, y).join(","))
                  .join(" ")}
              />
            )}
            {layers.show_pois &&
              activePois.map((poi) => {
                const [anchorX, anchorY] = worldToImagePixels(
                  map,
                  viewport,
                  poi.anchor_x,
                  poi.anchor_y,
                );
                const itemRef = refFromPoi(poi.poi_id);
                const itemKey = semanticKey(itemRef);
                const isSelected = selectedKey === itemKey;
                const coneLength = isSelected ? 58 : 0;
                const coneSpread = 0.34;
                return (
                  <g key={`${poi.poi_id}-overlay`}>
                    {coneLength > 0 && (
                      <path
                        className={isSelected ? "poi-view-cone is-selected" : "poi-view-cone"}
                        d={[
                          `M ${anchorX} ${anchorY}`,
                          `L ${anchorX + Math.cos(poi.anchor_yaw - coneSpread) * coneLength} ${
                            anchorY - Math.sin(poi.anchor_yaw - coneSpread) * coneLength
                          }`,
                          `A ${coneLength} ${coneLength} 0 0 1 ${
                            anchorX + Math.cos(poi.anchor_yaw + coneSpread) * coneLength
                          } ${
                            anchorY - Math.sin(poi.anchor_yaw + coneSpread) * coneLength
                          }`,
                          "Z",
                        ].join(" ")}
                      />
                    )}
                    <line
                      className="poi-heading"
                      x1={anchorX}
                      y1={anchorY}
                      x2={anchorX + Math.cos(poi.anchor_yaw) * 16}
                      y2={anchorY - Math.sin(poi.anchor_yaw) * 16}
                    />
                    {isSelected && (
                      <circle
                        className="poi-anchor-halo is-selected"
                        cx={anchorX}
                        cy={anchorY}
                        r={isSelected ? 10 : 8}
                      />
                    )}
                  </g>
                );
              })}
            {layers.show_yolo &&
              activeYoloObjects.map((object) => {
                const [x, y] = worldToImagePixels(map, viewport, object.world_x, object.world_y);
                const itemRef = refFromYoloObject(object.object_id);
                const itemKey = semanticKey(itemRef);
                const isSelected = selectedKey === itemKey;
                return (
                  <g key={`${object.object_id}-marker`}>
                    <circle
                      className={`yolo-ring ${isSelected ? "is-selected" : ""}`}
                      cx={x}
                      cy={y}
                      r={isSelected ? 11 : 9}
                    />
                    {isSelected && (
                      <circle
                        className="yolo-halo is-selected"
                        cx={x}
                        cy={y}
                        r={14}
                      />
                    )}
                    <circle
                      className={`yolo-dot ${isSelected ? "is-selected" : ""}`}
                      cx={x}
                      cy={y}
                      r={isSelected ? 6.5 : 5}
                    />
                  </g>
                );
              })}
            </svg>
            <div
              className="map-annotation-layer"
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: "100%",
                height: "100%",
              }}
            >
            {robotPose ? (
              <RobotOperatorMarkerView
                card={robotOperatorHoverCard}
                deployPop={false}
                gamecardDomId="map-robot-operator-gamecard"
                map={map}
                onPointerDownStop={stopEvent}
                operatorFleetGo2Hover={operatorFleetGo2Hover}
                stackIndex={0}
                viewport={viewport}
                worldX={robotPose.x}
                worldY={robotPose.y}
              />
            ) : null}
            {additionalRobotMarkers?.map((marker, index) => {
              const pos = additionalMarkerWorldPositions[index];
              if (!pos) {
                return null;
              }
              const [wx, wy] = pos;
              const safeId = marker.id.replace(/[^a-zA-Z0-9_-]/g, "_");
              return (
                <RobotOperatorMarkerView
                  key={marker.id}
                  card={marker.card}
                  deployPop={marker.deployPop === true}
                  gamecardDomId={`map-robot-operator-gamecard-${safeId}`}
                  map={map}
                  onPointerDownStop={stopEvent}
                  stackIndex={1 + index}
                  viewport={viewport}
                  worldX={wx}
                  worldY={wy}
                />
              );
            })}
            {layers.show_pois &&
              activePois.map((poi) => {
                const [x, y] = worldToImagePixels(map, viewport, poi.target_x, poi.target_y);
                const itemRef = refFromPoi(poi.poi_id);
                const itemKey = semanticKey(itemRef);
                const isSelected = selectedKey === itemKey;
                const cardImage = poi.hero_image_url || poi.thumbnail_url;
                const cardId = `poi-gamecard-${poi.poi_id.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
                const summary = poi.summary?.trim();
                return (
                  <button
                    className={["poi-pin", isSelected ? "is-selected" : ""]
                      .filter(Boolean)
                      .join(" ")}
                    aria-describedby={cardId}
                    aria-label={`POI ${poi.title}`}
                    key={poi.poi_id}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectItem(itemRef);
                    }}
                    onDragStart={preventNativeDrag}
                    onPointerDown={stopEvent}
                    style={{ left: `${x}px`, top: `${y}px` }}
                    type="button"
                  >
                    <img
                      alt=""
                      className="poi-pin-thumb"
                      decoding="async"
                      draggable={false}
                      onDragStart={preventNativeDrag}
                      src={poi.thumbnail_url}
                    />
                    <div className="poi-pin-gamecard" id={cardId} role="tooltip">
                      <div className="poi-pin-gamecard-media">
                        <img
                          alt=""
                          className="poi-pin-gamecard-img"
                          decoding="async"
                          draggable={false}
                          src={cardImage}
                        />
                      </div>
                      <div className="poi-pin-gamecard-body">
                        <div className="poi-pin-gamecard-title-row">
                          <span className="poi-pin-gamecard-title">{poi.title}</span>
                        </div>
                        {summary ? (
                          <p className="poi-pin-gamecard-summary">{summary}</p>
                        ) : null}
                        {poi.category ? (
                          <p className="poi-pin-gamecard-meta">
                            <span className="poi-pin-gamecard-meta-label">Category</span>{" "}
                            <span className="poi-pin-gamecard-meta-value">{poi.category}</span>
                          </p>
                        ) : null}
                        <p className="poi-pin-gamecard-meta">
                          <span className="poi-pin-gamecard-meta-label">Interest</span>{" "}
                          <span className="poi-pin-gamecard-meta-value">
                            {Math.round(poi.interest_score * 100)}%
                          </span>
                        </p>
                        {poi.objects.length > 0 ? (
                          <p className="poi-pin-gamecard-meta">
                            <span className="poi-pin-gamecard-meta-label">Objects</span>{" "}
                            <span className="poi-pin-gamecard-meta-value">
                              {poi.objects.slice(0, 4).join(", ")}
                              {poi.objects.length > 4 ? "…" : ""}
                            </span>
                          </p>
                        ) : null}
                      </div>
                    </div>
                  </button>
                );
              })}
            {layers.show_yolo &&
              activeYoloObjects.map((object) => {
                const [x, y] = worldToImagePixels(map, viewport, object.world_x, object.world_y);
                const itemRef = refFromYoloObject(object.object_id);
                const itemKey = semanticKey(itemRef);
                const isSelected = selectedKey === itemKey;
                return (
                  <button
                    className={["yolo-chip", "dot-only", isSelected ? "is-selected" : ""]
                      .filter(Boolean)
                      .join(" ")}
                    aria-label={`YOLO ${object.label}`}
                    key={object.object_id}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectItem(itemRef);
                    }}
                    onDragStart={preventNativeDrag}
                    onPointerDown={stopEvent}
                    style={{ left: `${x}px`, top: `${y}px` }}
                    title={`${object.label} ${Math.round(object.best_confidence * 100)}%`}
                    type="button"
                  >
                    <span className="sr-only">{object.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
            <div className="map-legend" onPointerDown={stopEvent}>
              <span>
                <i className="legend-swatch robot" />
                Robot
              </span>
              <span>
                <i className="legend-swatch path" />
                Planned path
              </span>
              <span>
                <i className="legend-swatch poi" />
                Point of Interest
              </span>
              <span>
                <i className="legend-swatch yolo" />
                YOLO objects
              </span>
            </div>
        </>
      )}
    </div>
  );
}
