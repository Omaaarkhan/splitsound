import type { ReactNode } from "react";
import "./Checkbox.css";

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  children: ReactNode;
  className?: string;
}

export function Checkbox({ checked, onChange, children, className }: CheckboxProps) {
  return (
    <label className={className ? `checkbox ${className}` : "checkbox"}>
      <input
        type="checkbox"
        className="checkbox__input"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span
        className={checked ? "checkbox__box checkbox__box--checked" : "checkbox__box"}
        aria-hidden="true"
      >
        {checked ? (
          <svg viewBox="0 0 12 12" className="checkbox__mark">
            <polyline points="2,6.5 4.8,9.2 10,3" />
          </svg>
        ) : null}
      </span>
      <span className="checkbox__label">{children}</span>
    </label>
  );
}
