import React from "react";

import {
  CameraIcon,
  InformationCircleIcon,
} from "@heroicons/react/24/outline";

import { ChatPanel } from "./ChatPanel";
import {
  EXAMPLE_AGENT_CHAT_MESSAGES,
  EXAMPLE_AGENT_SUGGESTION_PROMPTS,
} from "./exampleAgentChatMessages";
import { PanelShell } from "./PanelShell";
import { polarisDemoCameraCaptureSemanticItem } from "./semanticItems";
import { ChatState, SemanticItem, SemanticItemRef, SemanticKind } from "./types";

type ActivityEntry = {
  id: string;
  role: "operator" | "system";
  tone: "neutral" | "accent" | "success" | "danger";
  title: string;
  detail: string;
  timestamp: string;
};

type RailView = "timeline" | "semantic" | "chat";

export type SelectedSemanticPreview = {
  kind: SemanticKind;
  entity_id: string;
  title: string;
  subtitle: string;
  summary: string;
  thumbnail_url: string;
};

type OperatorRailProps = {
  activityEntries: ActivityEntry[];
  busyAction: string | null;
  chat: ChatState;
  items: SemanticItem[];
  selectedItem: SemanticItemRef | null;
  selectedPreview: SelectedSemanticPreview | null;
  onSelectItem: (item: SemanticItemRef | null) => void;
  onHighlightItem: (item: SemanticItemRef) => void;
  onFocusItem: (item: SemanticItemRef) => void;
  onGoToItem: (item: SemanticItemRef) => void;
  onClearFocus: () => void;
  onResetChat: () => void;
  onSubmitChatMessage: (message: string) => void;
  /** Omit outer `PanelShell` for sidebar card embedding */
  embedded?: boolean;
  /**
   * Navigator: `activity` = timeline only; `memory` = detection thumbnails only (no agent / detail card);
   * `agent` = chat only; `full` (default) = Timeline + Semantic + Orchestrator in one rail.
   */
  embedSegment?: "full" | "activity" | "memory" | "agent";
};

function itemLabel(kind: SemanticKind): string {
  return kind === "vlm_poi" ? "POI" : "YOLO";
}

function formatRelativeDetectionTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) {
    return "";
  }
  const sec = Math.floor((Date.now() - t) / 1000);
  if (sec < 45) {
    return "just now";
  }
  const min = Math.floor(sec / 60);
  if (min < 60) {
    return `${min} min${min === 1 ? "" : "s"} ago`;
  }
  const hr = Math.floor(min / 60);
  if (hr < 24) {
    return `${hr} hour${hr === 1 ? "" : "s"} ago`;
  }
  const day = Math.floor(hr / 24);
  if (day < 7) {
    return `${day} day${day === 1 ? "" : "s"} ago`;
  }
  const week = Math.floor(day / 7);
  if (week < 5) {
    return `${week} week${week === 1 ? "" : "s"} ago`;
  }
  return formatDetectionAbsDetail(iso);
}

function formatDetectionAbsDetail(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(d);
}

function formatDetectionDayHeader(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return "";
  }
  const parts = new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).formatToParts(d);
  const day = parts.find((p) => p.type === "day")?.value ?? "";
  const month = parts.find((p) => p.type === "month")?.value ?? "";
  const year = parts.find((p) => p.type === "year")?.value ?? "";
  if (day && month && year) {
    return `${day} ${month}, ${year}`;
  }
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(d);
}

function localDayKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return "unknown";
  }
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

function groupDetectionsByDay(itemsInput: SemanticItem[]): { header: string; items: SemanticItem[] }[] {
  const sorted = [...itemsInput].sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
  );
  const order: string[] = [];
  const byDay = new Map<string, SemanticItem[]>();
  for (const item of sorted) {
    const key = localDayKey(item.updated_at);
    if (!byDay.has(key)) {
      byDay.set(key, []);
      order.push(key);
    }
    byDay.get(key)!.push(item);
  }
  return order.map((key) => {
    const dayItems = byDay.get(key)!;
    const header = formatDetectionDayHeader(dayItems[0].updated_at);
    return { header, items: dayItems };
  });
}

export function OperatorRail(props: OperatorRailProps): React.ReactElement {
  const {
    activityEntries,
    busyAction,
    chat,
    items,
    selectedItem,
    selectedPreview,
    onSelectItem,
    onHighlightItem,
    onFocusItem,
    onGoToItem,
    onClearFocus,
    onResetChat,
    onSubmitChatMessage,
    embedded = false,
    embedSegment = "full",
  } = props;

  const segment = embedded ? embedSegment : "full";

  const [railView, setRailView] = React.useState<RailView>("timeline");
  const visibleEntries = React.useMemo(() => activityEntries.slice(-8), [activityEntries]);
  const demoDetectionItemRef = React.useRef<SemanticItem | null>(null);
  const detectionLogItems = React.useMemo(() => {
    if (items.length > 0) {
      demoDetectionItemRef.current = null;
      return items;
    }
    if (demoDetectionItemRef.current === null) {
      demoDetectionItemRef.current = polarisDemoCameraCaptureSemanticItem();
    }
    return [demoDetectionItemRef.current];
  }, [items]);
  const detectionDayGroups = React.useMemo(
    () => groupDetectionsByDay(detectionLogItems),
    [detectionLogItems],
  );

  const timelineEl = (
    <div className="rail-thread">
      {visibleEntries.map((entry) => (
        <article
          className={`thread-message role-${entry.role} tone-${entry.tone}`}
          key={entry.id}
        >
          <div className="thread-message-header">
            <strong>{entry.title}</strong>
            <time>{entry.timestamp}</time>
          </div>
          <p>{entry.detail}</p>
        </article>
      ))}
    </div>
  );

  const detectionLogEl = (
    <div aria-label="Detection history" className="polaris-detection-activity-log" role="feed">
      {detectionDayGroups.map((group) => (
        <section
          className="polaris-detection-activity-day"
          key={localDayKey(group.items[0].updated_at)}
        >
          <div className="polaris-detection-activity-day-label">{group.header}</div>
          <ul className="polaris-detection-activity-day-list">
            {group.items.map((item) => {
              const isYolo = item.kind === "yolo_object";
              const isSelected =
                selectedItem?.kind === item.kind && selectedItem?.entity_id === item.entity_id;
              const sourceLabel = isYolo ? "YOLO" : "VLM";
              const operatorName = "Unitree Go2";
              const action = isYolo
                ? `tagged a ${item.title.toLowerCase()}`
                : "took a new snapshot";
              const coords = `(${item.world_x.toFixed(2)}, ${item.world_y.toFixed(2)})`;
              return (
                <li key={`${item.kind}:${item.entity_id}`}>
                  <button
                    aria-pressed={isSelected}
                    className={[
                      "polaris-snapshot-alert",
                      isSelected ? "polaris-snapshot-alert--selected" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() =>
                      onSelectItem({ kind: item.kind, entity_id: item.entity_id })
                    }
                    type="button"
                  >
                    <span
                      className={[
                        "polaris-snapshot-alert-avatar",
                        isYolo
                          ? "polaris-snapshot-alert-avatar--yolo"
                          : "polaris-snapshot-alert-avatar--vlm",
                      ].join(" ")}
                    >
                      {item.thumbnail_url ? (
                        <img
                          alt=""
                          className="polaris-snapshot-alert-avatar-img"
                          decoding="async"
                          src={item.thumbnail_url}
                        />
                      ) : (
                        <CameraIcon
                          aria-hidden
                          className="polaris-snapshot-alert-avatar-fallback"
                        />
                      )}
                    </span>
                    <span className="polaris-snapshot-alert-body">
                      <span className="polaris-snapshot-alert-title">
                        <strong>{operatorName}</strong> {action} for a POI{" "}
                        <span className="polaris-snapshot-alert-coords">@ {coords}</span>
                      </span>
                      <span className="polaris-snapshot-alert-desc">
                        <span className="polaris-snapshot-alert-source">{sourceLabel}</span>
                        <span className="polaris-snapshot-alert-dot" aria-hidden>
                          ·
                        </span>
                        <span className="polaris-snapshot-alert-rel">
                          {formatRelativeDetectionTime(item.updated_at)}
                        </span>
                        <span className="polaris-snapshot-alert-dot" aria-hidden>
                          ·
                        </span>
                        <time
                          className="polaris-snapshot-alert-abs"
                          dateTime={item.updated_at}
                        >
                          {formatDetectionAbsDetail(item.updated_at)}
                        </time>
                      </span>
                    </span>
                    <InformationCircleIcon
                      aria-hidden
                      className="polaris-snapshot-alert-icon"
                    />
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );

  const semanticListEl =
    items.length > 0 ? (
      <div className="rail-poi-list">
        {items.map((item) => {
          const isSelected =
            selectedItem?.kind === item.kind && selectedItem?.entity_id === item.entity_id;
          return (
            <article className="rail-poi-item" key={`${item.kind}:${item.entity_id}`}>
              <button
                className={`rail-poi-main ${isSelected ? "is-active" : ""}`}
                onClick={() => onSelectItem({ kind: item.kind, entity_id: item.entity_id })}
                type="button"
              >
                <img alt={item.title} src={item.thumbnail_url} />
                <span className="rail-poi-copy">
                  <strong>{item.title}</strong>
                  <span>
                    {itemLabel(item.kind)} · {item.subtitle}
                  </span>
                </span>
              </button>
              <button
                className="mini-button mini-button-primary"
                disabled={busyAction === `go-${item.kind}-${item.entity_id}`}
                onClick={() => onGoToItem({ kind: item.kind, entity_id: item.entity_id })}
                type="button"
              >
                Go
              </button>
            </article>
          );
        })}
      </div>
    ) : (
      <div className="thread-empty">
        {segment === "memory" ? "No detections yet." : "No semantic anchors yet."}
      </div>
    );

  const selectedCard = selectedPreview ? (
    <section className="rail-selected-card">
      <img alt={selectedPreview.title} src={selectedPreview.thumbnail_url} />
      <div className="rail-selected-copy">
        <div className="rail-selected-topline">
          <strong>{selectedPreview.title}</strong>
          <span>{selectedPreview.subtitle}</span>
        </div>
        <p>{selectedPreview.summary}</p>
      </div>
      <div className="rail-selected-actions">
        <button
          className="mini-button"
          onClick={() => onFocusItem({ kind: selectedPreview.kind, entity_id: selectedPreview.entity_id })}
          type="button"
        >
          Focus
        </button>
        <button
          className="mini-button"
          onClick={() =>
            onHighlightItem({ kind: selectedPreview.kind, entity_id: selectedPreview.entity_id })
          }
          type="button"
        >
          Highlight
        </button>
        <button
          className="mini-button mini-button-primary"
          disabled={busyAction === `go-${selectedPreview.kind}-${selectedPreview.entity_id}`}
          onClick={() => onGoToItem({ kind: selectedPreview.kind, entity_id: selectedPreview.entity_id })}
          type="button"
        >
          Go
        </button>
        <button className="mini-button" onClick={() => onClearFocus()} type="button">
          Clear
        </button>
      </div>
    </section>
  ) : null;

  const tabsFull = (
    <div className="rail-tabs" role="tablist" aria-label="Operator rail views">
      <button
        className={railView === "timeline" ? "is-active" : ""}
        onClick={() => {
          setRailView("timeline");
        }}
        type="button"
      >
        Timeline
      </button>
      <button
        className={railView === "semantic" ? "is-active" : ""}
        onClick={() => {
          setRailView("semantic");
        }}
        type="button"
      >
        Semantic
      </button>
      <button
        className={railView === "chat" ? "is-active" : ""}
        onClick={() => {
          setRailView("chat");
        }}
        type="button"
      >
        Orchestrator
      </button>
    </div>
  );

  if (embedded && segment === "activity") {
    return (
      <div className="operator-rail-embedded operator-rail-embedded--segment-activity">
        <div className="rail-stack">
          <div className="rail-content">{timelineEl}</div>
        </div>
      </div>
    );
  }

  if (embedded && segment === "memory") {
    return (
      <div className="operator-rail-embedded operator-rail-embedded--segment-memory">
        <div className="rail-stack">
          <div className="rail-content">{detectionLogEl}</div>
        </div>
      </div>
    );
  }

  if (embedded && segment === "agent") {
    return (
      <div className="operator-rail-embedded operator-rail-embedded--segment-agent">
        <div className="rail-stack">
          <div className="rail-content">
            <div className="polaris-navigator-orchestrator-chat-deferred">
              <ChatPanel
                chat={chat}
                exampleSuggestionPrompts={EXAMPLE_AGENT_SUGGESTION_PROMPTS}
                exampleWhenEmpty={EXAMPLE_AGENT_CHAT_MESSAGES}
                onResetChat={onResetChat}
                onSubmitMessage={onSubmitChatMessage}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  const body = (
    <div className="rail-stack">
      {selectedCard}
      <div className="rail-content">
        {railView === "timeline" ? (
          timelineEl
        ) : railView === "semantic" ? (
          semanticListEl
        ) : (
          <ChatPanel
            chat={chat}
            exampleSuggestionPrompts={EXAMPLE_AGENT_SUGGESTION_PROMPTS}
            exampleWhenEmpty={EXAMPLE_AGENT_CHAT_MESSAGES}
            onResetChat={onResetChat}
            onSubmitMessage={onSubmitChatMessage}
          />
        )}
      </div>
    </div>
  );

  if (embedded) {
    return (
      <div className="operator-rail-embedded">
        <div className="operator-rail-embedded-tabs">{tabsFull}</div>
        {body}
      </div>
    );
  }

  return (
    <PanelShell
      aside={tabsFull}
      bodyClassName="panel-body-console"
      className="console-panel"
    >
      {body}
    </PanelShell>
  );
}
