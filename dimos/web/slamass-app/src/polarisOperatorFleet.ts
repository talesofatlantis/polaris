import {
  POLARIS_GO2_EDU_PREVIEW_URL,
  POLARIS_GO2_PREVIEW_URL,
  POLARIS_OPERATOR_MOUNT_GO2_EDU_URL,
  POLARIS_OPERATOR_SELECT_THUMB_URL,
} from "./polarisAssets";

export type PolarisOperatorFleetEntry = {
  id: string;
  title: string;
  titleHref?: string;
  category?: { label: string; value: string };
  location: string;
  task: string;
  /** Optional live battery state-of-charge (0–100). Null/undefined → render as "—". */
  batteryPercent?: number | null;
  imageUrl: string | null;
  imageAlt: string;
  active?: "green" | "blue" | "grey";
  emptyVisualLabel?: string;
  mountThumbUrl: string | null;
  mountValue?: string;
  mountThumbHref?: string;
};

/** Shared roster for `/polaris/operators` and navigator sidebar. */
export const POLARIS_OPERATOR_FLEET: PolarisOperatorFleetEntry[] = [
  {
    id: "go2",
    title: "Unitree Go2",
    category: { label: "Type", value: "Unitree Go2 X" },
    location: "Floor lab",
    task: "Patrol",
    active: "green",
    imageUrl: POLARIS_GO2_PREVIEW_URL,
    imageAlt: "Unitree Go2 robot",
    mountThumbUrl: POLARIS_OPERATOR_SELECT_THUMB_URL,
  },
  {
    id: "go2-platform",
    title: "Unitree Go2",
    category: { label: "Type", value: "Unitree Go2 EDU" },
    location: "Bench",
    task: "Calibration",
    active: "grey",
    imageUrl: POLARIS_GO2_EDU_PREVIEW_URL,
    imageAlt: "Unitree Go2",
    mountThumbUrl: POLARIS_OPERATOR_MOUNT_GO2_EDU_URL,
    mountValue: "D1-T",
    mountThumbHref: "https://www.unitree.com/D1-T",
  },
];
