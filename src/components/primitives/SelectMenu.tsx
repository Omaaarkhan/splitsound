import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import "./SelectMenu.css";

export interface SelectOption<T extends string | number> {
  value: T;
  label: string;
}

interface SelectMenuProps<T extends string | number> {
  value: T;
  options: readonly SelectOption<T>[];
  label: string;
  onChange: (value: T) => void;
  className?: string;
}

export function SelectMenu<T extends string | number>({
  value,
  options,
  label,
  onChange,
  className,
}: SelectMenuProps<T>) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(() =>
    Math.max(
      0,
      options.findIndex((o) => o.value === value),
    ),
  );
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listId = useId();

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  const selected = options.find((o) => o.value === value) ?? options[0];

  const openMenu = () => {
    const idx = Math.max(
      0,
      options.findIndex((o) => o.value === value),
    );
    setActiveIndex(idx);
    setOpen(true);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openMenu();
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setActiveIndex((i) => Math.min(options.length - 1, i + 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setActiveIndex((i) => Math.max(0, i - 1));
        break;
      case "Home":
        e.preventDefault();
        setActiveIndex(0);
        break;
      case "End":
        e.preventDefault();
        setActiveIndex(options.length - 1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (options[activeIndex]) {
          onChange(options[activeIndex].value);
        }
        setOpen(false);
        break;
      case "Escape":
        e.preventDefault();
        e.stopPropagation();
        setOpen(false);
        break;
    }
  };

  return (
    <div ref={rootRef} className={className ? `select-menu ${className}` : "select-menu"}>
      <button
        type="button"
        className="select-menu__trigger"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        onClick={() => (open ? setOpen(false) : openMenu())}
        onKeyDown={handleKeyDown}
      >
        <span>{selected?.label ?? ""}</span>
        <span className="select-menu__caret" aria-hidden="true" />
      </button>
      {open ? (
        <ul id={listId} role="listbox" aria-label={label} className="select-menu__list">
          {options.map((opt, idx) => {
            const isSelected = opt.value === value;
            const isActive = idx === activeIndex;
            const classes = [
              "select-menu__option",
              isActive ? "select-menu__option--active" : "",
              isSelected ? "select-menu__option--selected" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <li
                key={String(opt.value)}
                role="option"
                aria-selected={isSelected}
                className={classes}
                onMouseEnter={() => setActiveIndex(idx)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(opt.value);
                  setOpen(false);
                }}
              >
                {opt.label}
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
