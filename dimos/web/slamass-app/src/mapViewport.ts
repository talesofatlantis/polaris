import type { AppState, MapState, RobotPose, UiCameraState } from "./types";

/** Below 1 shows more map (zoomed out); above 1 magnifies. */
export const MIN_MAP_ZOOM = 0.25;
export const MAX_MAP_ZOOM = 4;
/**
 * Overview zoom after fit (centered on map). zoom=1 touches the shorter view axis; lower = more margin.
 * Must stay aligned with `UI_DEFAULT_MAP_ZOOM` in `dimos/slamass/service.py`.
 */
export const DEFAULT_MAP_ZOOM = 0.78;

export type MapViewport = {
  width: number;
  height: number;
  pixelsPerCell: number;
  centerCellX: number;
  centerCellY: number;
  imageLeft: number;
  imageTop: number;
  imageWidth: number;
  imageHeight: number;
  /** 0–1: horizontal anchor where the camera center is placed on screen (default 0.5). */
  screenAnchorX: number;
  /** 0–1: vertical anchor where the camera center is placed on screen (default 0.5). Lower = map sits higher. */
  screenAnchorY: number;
};

/** Framing passed to `buildViewport`; values outside ~0.05–0.95 are clamped. */
export type BuildViewportOptions = {
  screenAnchorX?: number;
  screenAnchorY?: number;
};

/**
 * Default vertical anchor for Navigator / Polaris map column: camera center sits above the geometric
 * middle so the costmap reads higher under the panel header (tall map stage).
 */
export const NAVIGATOR_MAP_VIEWPORT_ANCHOR_Y = 0.38;

function clampScreenAnchor(value: number): number {
  return Math.min(0.95, Math.max(0.05, value));
}

export function clampZoom(zoom: number): number {
  return Math.max(MIN_MAP_ZOOM, Math.min(MAX_MAP_ZOOM, zoom));
}

export function getMapCenter(map: MapState): [number, number] {
  return [
    map.origin_x + (map.width * map.resolution) / 2,
    map.origin_y + (map.height * map.resolution) / 2,
  ];
}

/** Keep synthetic markers slightly inside the costmap so they stay visible after zoom/pan. */
export function clampWorldToMapBounds(map: MapState, x: number, y: number): [number, number] {
  const pad = map.resolution * 6;
  const minX = map.origin_x + pad;
  const maxX = map.origin_x + map.width * map.resolution - pad;
  const minY = map.origin_y + pad;
  const maxY = map.origin_y + map.height * map.resolution - pad;
  if (minX >= maxX || minY >= maxY) {
    const [cx, cy] = getMapCenter(map);
    return [cx, cy];
  }
  return [
    Math.min(Math.max(x, minX), maxX),
    Math.min(Math.max(y, minY), maxY),
  ];
}

/**
 * World pose for an extra operator marker (e.g. newly deployed). Offsets from the live robot when
 * available; otherwise from map center.
 */
export function worldPositionForAdditionalOperator(
  map: MapState,
  robotPose: RobotPose | null,
  index: number,
): [number, number] {
  const [cx, cy] = getMapCenter(map);
  if (robotPose) {
    const baseAngle = robotPose.yaw + Math.PI / 2;
    const angle = baseAngle + index * 0.52;
    const dist = 1.05 + index * 0.42;
    return clampWorldToMapBounds(
      map,
      robotPose.x + Math.cos(angle) * dist,
      robotPose.y + Math.sin(angle) * dist,
    );
  }
  const ox = 1.0 + index * 0.35;
  const oy = 0.55 + index * 0.22;
  return clampWorldToMapBounds(map, cx + ox, cy + oy);
}

/** Same framing as POST /api/ui/focus-map: map center + overview zoom. */
export function fitOverviewCamera(map: MapState): UiCameraState {
  const [cx, cy] = getMapCenter(map);
  return normalizeCamera(map, {
    center_x: cx,
    center_y: cy,
    zoom: DEFAULT_MAP_ZOOM,
  });
}

export function applyFitOverviewCameraToAppState(state: AppState): AppState {
  if (!state.map) {
    return state;
  }
  return {
    ...state,
    ui: {
      ...state.ui,
      camera: fitOverviewCamera(state.map),
    },
  };
}

export function normalizeCamera(map: MapState, camera: UiCameraState): UiCameraState {
  const [defaultCenterX, defaultCenterY] = getMapCenter(map);
  const minX = map.origin_x;
  const maxX = map.origin_x + map.width * map.resolution;
  const minY = map.origin_y;
  const maxY = map.origin_y + map.height * map.resolution;

  const centerX = camera.center_x ?? defaultCenterX;
  const centerY = camera.center_y ?? defaultCenterY;

  return {
    center_x: Math.min(Math.max(centerX, minX), maxX),
    center_y: Math.min(Math.max(centerY, minY), maxY),
    zoom: clampZoom(camera.zoom),
  };
}

export function buildViewport(
  map: MapState,
  width: number,
  height: number,
  camera: UiCameraState,
  options?: BuildViewportOptions,
): MapViewport {
  const screenAnchorX = clampScreenAnchor(options?.screenAnchorX ?? 0.5);
  const screenAnchorY = clampScreenAnchor(options?.screenAnchorY ?? 0.5);
  const normalizedCamera = normalizeCamera(map, camera);
  const basePixelsPerCell = Math.min(width / map.width, height / map.height);
  const pixelsPerCell = basePixelsPerCell * normalizedCamera.zoom;
  const centerCellX = (normalizedCamera.center_x! - map.origin_x) / map.resolution;
  const centerCellY = (normalizedCamera.center_y! - map.origin_y) / map.resolution;
  const imageWidth = map.width * pixelsPerCell;
  const imageHeight = map.height * pixelsPerCell;

  return {
    width,
    height,
    pixelsPerCell,
    centerCellX,
    centerCellY,
    imageLeft: width * screenAnchorX - centerCellX * pixelsPerCell,
    imageTop: height * screenAnchorY - (map.height - centerCellY) * pixelsPerCell,
    imageWidth,
    imageHeight,
    screenAnchorX,
    screenAnchorY,
  };
}

export function worldToScreen(
  map: MapState,
  viewport: MapViewport,
  x: number,
  y: number,
): [number, number] {
  const cellX = (x - map.origin_x) / map.resolution;
  const cellY = (y - map.origin_y) / map.resolution;
  const imageX = cellX * viewport.pixelsPerCell;
  const imageY = (map.height - cellY) * viewport.pixelsPerCell;
  return [viewport.imageLeft + imageX, viewport.imageTop + imageY];
}

export function worldToImagePixels(
  map: MapState,
  viewport: MapViewport,
  x: number,
  y: number,
): [number, number] {
  const cellX = (x - map.origin_x) / map.resolution;
  const cellY = (y - map.origin_y) / map.resolution;
  return [
    cellX * viewport.pixelsPerCell,
    (map.height - cellY) * viewport.pixelsPerCell,
  ];
}

export function screenToWorld(
  map: MapState,
  viewport: MapViewport,
  screenX: number,
  screenY: number,
): [number, number] {
  const cellX = (screenX - viewport.imageLeft) / viewport.pixelsPerCell;
  const cellY = map.height - (screenY - viewport.imageTop) / viewport.pixelsPerCell;
  return [
    map.origin_x + cellX * map.resolution,
    map.origin_y + cellY * map.resolution,
  ];
}

export function panCamera(
  map: MapState,
  camera: UiCameraState,
  deltaX: number,
  deltaY: number,
  viewport: MapViewport,
): UiCameraState {
  const normalized = normalizeCamera(map, camera);
  const deltaWorldX = (-deltaX / viewport.pixelsPerCell) * map.resolution;
  const deltaWorldY = (deltaY / viewport.pixelsPerCell) * map.resolution;
  return normalizeCamera(map, {
    center_x: normalized.center_x! + deltaWorldX,
    center_y: normalized.center_y! + deltaWorldY,
    zoom: normalized.zoom,
  });
}

export function zoomCameraAtScreenPoint(
  map: MapState,
  camera: UiCameraState,
  screenX: number,
  screenY: number,
  nextZoom: number,
  viewport: MapViewport,
): UiCameraState {
  const clampedZoom = clampZoom(nextZoom);
  const [worldX, worldY] = screenToWorld(map, viewport, screenX, screenY);
  const cellX = (worldX - map.origin_x) / map.resolution;
  const cellY = (worldY - map.origin_y) / map.resolution;
  const basePixelsPerCell = Math.min(viewport.width / map.width, viewport.height / map.height);
  const pixelsPerCell = basePixelsPerCell * clampedZoom;
  const ax = viewport.width * viewport.screenAnchorX;
  const ay = viewport.height * viewport.screenAnchorY;
  const centerCellX = cellX - (screenX - ax) / pixelsPerCell;
  const centerCellY = cellY + (screenY - ay) / pixelsPerCell;

  return normalizeCamera(map, {
    center_x: map.origin_x + centerCellX * map.resolution,
    center_y: map.origin_y + centerCellY * map.resolution,
    zoom: clampedZoom,
  });
}
