import {
  ArrowsPointingInIcon,
  ArrowsPointingOutIcon,
} from "@heroicons/react/24/outline";
import React, { useCallback, useEffect, useId, useRef, useState } from "react";

import {
  POLARIS_OPERATOR_FLEET,
  type PolarisOperatorFleetEntry,
} from "../polarisOperatorFleet";
import { NavigatorOptionCard } from "./NavigatorOptionCard";

function statusLabel(active: "green" | "blue" | "grey"): string {
  if (active === "blue") {
    return "Standby";
  }
  if (active === "grey") {
    return "Inactive";
  }
  return "Active";
}

function NavigatorOperatorFleetList(props: {
  onGo2OperatorHoverChange?: (hovered: boolean) => void;
  prependedOperators?: PolarisOperatorFleetEntry[];
  hideDemoFleet?: boolean;
}): React.ReactElement {
  const {
    onGo2OperatorHoverChange,
    prependedOperators = [],
    hideDemoFleet = false,
  } = props;
  const roster = hideDemoFleet
    ? prependedOperators
    : [...prependedOperators, ...POLARIS_OPERATOR_FLEET];

  // Empty state — no live operator and no deployed entries from create flow.
  // Show a clean prompt to add one instead of the demo placeholder fleet.
  if (roster.length === 0) {
    return (
      <div className="polaris-nav-operators-empty">
        <p className="polaris-nav-operators-empty-title">No operators connected</p>
        <p className="polaris-nav-operators-empty-sub">
          Plug a Go2 in via Ethernet, or set up a fresh operator from the Polaris
          shell.
        </p>
        <a
          className="polaris-nav-operators-empty-cta"
          href="/polaris/create"
        >
          Add operator
        </a>
      </div>
    );
  }
  return (
    <div className="polaris-nav-operators-embed-shell">
      <ul
        aria-label="Operators"
        className="polaris-operators-list polaris-nav-operators-embed"
        role="list"
      >
        {roster.map((op) => {
          const go2HoverHandlers =
            op.id === "go2" && onGo2OperatorHoverChange
              ? {
                  onMouseEnter: () => {
                    onGo2OperatorHoverChange(true);
                  },
                  onMouseLeave: () => {
                    onGo2OperatorHoverChange(false);
                  },
                }
              : {};
          return (
            <li className="polaris-operators-list-item" key={op.id} role="listitem">
              <article
                aria-label={op.title}
                className="polaris-operator-card polaris-nav-operators-embed-card"
                data-testid={`polaris-nav-operator-${op.id}`}
                {...go2HoverHandlers}
              >
                <div className="polaris-operator-card-media-column">
                  <div
                    className={
                      op.imageUrl
                        ? "polaris-operator-card-visual"
                        : "polaris-operator-card-visual polaris-operator-card-visual--empty"
                    }
                  >
                    {op.imageUrl ? (
                      <img
                        alt={op.imageAlt}
                        className="polaris-operator-card-img polaris-operator-card-img--static"
                        decoding="async"
                        src={op.imageUrl}
                      />
                    ) : (
                      <span className="polaris-operator-card-placeholder">
                        {op.emptyVisualLabel ?? "Preview"}
                      </span>
                    )}
                  </div>
                </div>
                <div className="polaris-operator-card-body">
                  <h2 className="polaris-operator-card-title">
                    <span className="polaris-operator-card-title-text">
                      {op.titleHref ? (
                        <a
                          className="polaris-operator-card-title-link"
                          href={op.titleHref}
                          rel="noopener noreferrer"
                          target="_blank"
                        >
                          {op.title}
                        </a>
                      ) : (
                        op.title
                      )}
                    </span>
                    {op.active ? (
                      <span
                        aria-label={statusLabel(op.active)}
                        className="polaris-operator-card-active"
                        role="status"
                      >
                        <span
                          aria-hidden
                          className={
                            op.active === "blue"
                              ? "polaris-operator-card-active-dot polaris-operator-card-active-dot--blue"
                              : op.active === "grey"
                                ? "polaris-operator-card-active-dot polaris-operator-card-active-dot--grey"
                                : "polaris-operator-card-active-dot polaris-operator-card-active-dot--green"
                          }
                        />
                      </span>
                    ) : null}
                  </h2>
                  {op.category ? (
                    <p className="polaris-operator-card-sub polaris-operator-card-meta">
                      <span className="polaris-operator-card-meta-label">{op.category.label}</span>{" "}
                      <span className="polaris-operator-card-meta-value">{op.category.value}</span>
                    </p>
                  ) : null}
                  <p className="polaris-operator-card-sub polaris-operator-card-meta">
                    <span className="polaris-operator-card-meta-label">Location</span>{" "}
                    <span className="polaris-operator-card-meta-value">{op.location}</span>
                  </p>
                  <p className="polaris-operator-card-sub polaris-operator-card-meta">
                    <span className="polaris-operator-card-meta-label">Mission</span>{" "}
                    <span className="polaris-operator-card-meta-value">{op.task}</span>
                  </p>
                  {op.batteryPercent !== undefined ? (
                    <p className="polaris-operator-card-sub polaris-operator-card-meta polaris-operator-card-meta--battery">
                      <span className="polaris-operator-card-meta-label">
                        Battery
                      </span>{" "}
                      <span className="polaris-operator-card-meta-value">
                        {op.batteryPercent === null
                          ? "—"
                          : `${op.batteryPercent}%`}
                      </span>
                    </p>
                  ) : null}
                </div>
              </article>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export type NavigatorOperatorFleetProps = {
  /** First fleet row (Unitree Go2) drives the map robot gamecard hover when set. */
  onGo2OperatorHoverChange?: (hovered: boolean) => void;
  /** Operators from a completed create flow; listed above the demo fleet. */
  prependedOperators?: PolarisOperatorFleetEntry[];
  /** When a real robot is connected, suppress the static demo fleet so only live operators show. */
  hideDemoFleet?: boolean;
};

/**
 * Fleet roster in the navigator “Operators” column — same card structure as `/polaris/operators`, scaled for the sidebar.
 */
export function NavigatorOperatorFleet(
  props: NavigatorOperatorFleetProps,
): React.ReactElement {
  const { onGo2OperatorHoverChange, prependedOperators, hideDemoFleet } = props;
  const [extended, setExtended] = useState(false);
  const titleId = useId();
  const collapseButtonRef = useRef<HTMLButtonElement>(null);

  const closeExtended = useCallback(() => {
    setExtended(false);
  }, []);

  useEffect(() => {
    if (!extended) {
      return;
    }
    const id = window.requestAnimationFrame(() => {
      collapseButtonRef.current?.focus();
    });
    return () => {
      window.cancelAnimationFrame(id);
    };
  }, [extended]);

  useEffect(() => {
    if (!extended) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [extended]);

  useEffect(() => {
    if (!extended) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeExtended();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [extended, closeExtended]);

  return (
    <>
      <NavigatorOptionCard
        className="polaris-nav-operators-fleet-card polaris-nav-option-card--polaris-heading"
        headerAside={
          <div className="polaris-nav-operators-header-aside">
            <button
              aria-expanded={extended}
              aria-label="Expand operators"
              className="polaris-nav-operators-extend-btn"
              data-testid="polaris-nav-operators-expand"
              type="button"
              onClick={() => {
                setExtended(true);
              }}
            >
              <ArrowsPointingOutIcon
                aria-hidden
                className="polaris-nav-operators-extend-btn-icon"
              />
            </button>
          </div>
        }
        title="Operators"
      >
        <NavigatorOperatorFleetList
          hideDemoFleet={hideDemoFleet}
          onGo2OperatorHoverChange={onGo2OperatorHoverChange}
          prependedOperators={prependedOperators}
        />
      </NavigatorOptionCard>

      {extended ? (
        <div
          aria-labelledby={titleId}
          aria-modal="true"
          className="polaris-nav-operators-extend-overlay"
          role="dialog"
        >
          <button
            aria-label="Close expanded operators"
            className="polaris-nav-operators-extend-backdrop"
            type="button"
            onClick={closeExtended}
          />
          <div className="polaris-nav-operators-extend-panel">
            <header className="polaris-nav-operators-extend-panel-head">
              <h2 className="polaris-nav-operators-extend-panel-title" id={titleId}>
                Operators
              </h2>
              <button
                ref={collapseButtonRef}
                aria-label="Collapse operators"
                className="polaris-nav-operators-extend-btn"
                data-testid="polaris-nav-operators-collapse"
                type="button"
                onClick={closeExtended}
              >
                <ArrowsPointingInIcon
                  aria-hidden
                  className="polaris-nav-operators-extend-btn-icon"
                />
              </button>
            </header>
            <div className="polaris-nav-operators-extend-panel-body">
              <NavigatorOperatorFleetList
                onGo2OperatorHoverChange={onGo2OperatorHoverChange}
                prependedOperators={prependedOperators}
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
