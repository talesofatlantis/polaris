export type SemanticKind = "vlm_poi" | "yolo_object";
export type ManualInspectionMode = "ai_gate" | "always_create";
export type YoloRuntimeMode = "live" | "paused";

export interface RobotPose {
  x: number;
  y: number;
  z: number;
  yaw: number;
}

export interface SemanticItemRef {
  kind: SemanticKind;
  entity_id: string;
}

export interface PovState {
  available: boolean;
  seq: number;
  updated_at: string | null;
  image_url: string;
}

export interface MapState {
  map_id: string;
  resolution: number;
  origin_x: number;
  origin_y: number;
  width: number;
  height: number;
  updated_at: string;
  image_version: number;
  image_url: string;
}

export interface UiCameraState {
  center_x: number | null;
  center_y: number | null;
  zoom: number;
}

export interface UiState {
  revision: number;
  camera: UiCameraState;
  selected_item: SemanticItemRef | null;
  highlighted_items: SemanticItemRef[];
}

export interface Poi {
  poi_id: string;
  map_id: string;
  anchor_x: number;
  anchor_y: number;
  anchor_yaw: number;
  target_x: number;
  target_y: number;
  title: string;
  summary: string;
  category: string;
  interest_score: number;
  status: string;
  objects: string[];
  created_at: string;
  updated_at: string;
  thumbnail_url: string;
  hero_image_url: string;
}

export interface YoloObject {
  object_id: string;
  map_id: string;
  label: string;
  class_id: number;
  world_x: number;
  world_y: number;
  world_z: number;
  size_x: number;
  size_y: number;
  size_z: number;
  best_view_x: number;
  best_view_y: number;
  best_view_yaw: number;
  status: string;
  detections_count: number;
  best_confidence: number;
  created_at: string;
  updated_at: string;
  last_seen_at: string;
  thumbnail_url: string;
  hero_image_url: string;
}

export interface SemanticItem {
  kind: SemanticKind;
  entity_id: string;
  title: string;
  subtitle: string;
  world_x: number;
  world_y: number;
  world_yaw: number;
  thumbnail_url: string;
  updated_at: string;
  /** Optional POI summary for detection-log detail line */
  summary?: string;
}

export interface InspectionState {
  status: string;
  message: string;
  poi_id: string | null;
}

export interface InspectionSettings {
  manual_mode: ManualInspectionMode;
}

export interface YoloRuntimeState {
  mode: YoloRuntimeMode;
  inference_enabled: boolean;
}

export interface LayerVisibility {
  show_pois: boolean;
  show_yolo: boolean;
}

export interface ChatMessage {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  status: string;
  tools_used: string[];
}

export interface ChatState {
  running: boolean;
  messages: ChatMessage[];
}

export interface ChatToolParameter {
  name: string;
  type: string;
  required: boolean;
  description?: string;
  enum?: string[];
  item_type?: string;
}

export interface ChatToolDefinition {
  name: string;
  description: string;
  parameters: ChatToolParameter[];
}

export interface AppState {
  /** websocket_vis base (Socket.IO + dashboard on `/` when using `rerun-web`). From slamass `--map-socket-url`. */
  dimos_viewer_url: string | null;
  /** Rerun 3D web viewer (`:9090` → gRPC `:9876`), same as DimOS dashboard right pane. Requires `dimos --viewer rerun-web run …`. */
  dimos_rerun_web_viewer_url: string | null;
  /** When false, VLM features use fallbacks: Inspect still saves a POI from the camera; agent chat uses a stub backend. */
  openai_configured: boolean;
  connected: boolean;
  /** Battery state-of-charge from the Go2 (0–100). Null until firmware reports a recognised field. */
  battery_percent: number | null;
  robot_pose: RobotPose | null;
  path: Array<[number, number]>;
  pov: PovState;
  map: MapState | null;
  pois: Poi[];
  yolo_objects: YoloObject[];
  inspection: InspectionState;
  inspection_settings: InspectionSettings;
  yolo_runtime: YoloRuntimeState;
  layers: LayerVisibility;
  ui: UiState;
  chat: ChatState;
}
