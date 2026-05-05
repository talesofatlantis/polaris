import React, { useId } from "react";

type PolarisIconSwitchProps = {
  /** Current on/off state. */
  checked: boolean;
  /** Called with the next state. Both label clicks and the track itself fire this. */
  onCheckedChange: (next: boolean) => void;
  /** Icon shown on the left (active when `checked === false`). */
  offIcon: React.ReactNode;
  /** Icon shown on the right (active when `checked === true`). */
  onIcon: React.ReactNode;
  /** Optional aria description (e.g. "Toggle searchlight"). */
  ariaLabel?: string;
  /** Optional disabled state — track + both labels become unclickable, dimmed. */
  disabled?: boolean;
  /** Hides the inactive label slightly, like shadcn switch-11. Defaults to true. */
  fadeInactive?: boolean;
  /** Extra class on the wrapping group. */
  className?: string;
};

/**
 * Dual-icon-label switch — Polaris flavour of shadcn `switch-11`.
 *
 * Layout: [offIcon] · [track + thumb] · [onIcon]
 *
 * Behaviour:
 * - Clicking `offIcon` forces `checked = false`.
 * - Clicking `onIcon` forces `checked = true`.
 * - Clicking the track toggles.
 * - The inactive icon dims to `--text-muted` (matching shadcn's
 *   `group-data-[state=…]:text-muted-foreground/70` pattern).
 *
 * The visual chrome (rounded pill track, animated thumb, focus ring) follows
 * the Polaris card aesthetic — slate hairlines, white surface, soft amber for
 * the ON glow so it reads as "the light is shining."
 */
export function PolarisIconSwitch(props: PolarisIconSwitchProps): React.ReactElement {
  const {
    checked,
    onCheckedChange,
    offIcon,
    onIcon,
    ariaLabel,
    disabled = false,
    fadeInactive = true,
    className,
  } = props;
  const id = useId();
  const offId = `${id}-off`;
  const onId = `${id}-on`;
  const dataState = checked ? "checked" : "unchecked";

  return (
    <span
      className={[
        "polaris-icon-switch",
        fadeInactive ? "polaris-icon-switch--fade-inactive" : "",
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
      data-state={dataState}
    >
      <button
        type="button"
        id={offId}
        aria-controls={id}
        aria-label="Off"
        className="polaris-icon-switch-label polaris-icon-switch-label--off"
        disabled={disabled}
        onClick={() => onCheckedChange(false)}
      >
        {offIcon}
      </button>
      <button
        type="button"
        id={id}
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${onId} ${offId}`}
        aria-label={ariaLabel}
        className="polaris-icon-switch-track"
        data-state={dataState}
        disabled={disabled}
        onClick={() => onCheckedChange(!checked)}
      >
        <span aria-hidden className="polaris-icon-switch-thumb" />
      </button>
      <button
        type="button"
        id={onId}
        aria-controls={id}
        aria-label="On"
        className="polaris-icon-switch-label polaris-icon-switch-label--on"
        disabled={disabled}
        onClick={() => onCheckedChange(true)}
      >
        {onIcon}
      </button>
    </span>
  );
}
