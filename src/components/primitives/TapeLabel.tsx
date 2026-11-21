import type { DragEvent } from "react";
import type { StemId } from "../../ipc/contract";
import { stemName } from "../../utils/stems";
import "./TapeLabel.css";

interface TapeLabelProps {
  stemId: StemId;
  size?: "default" | "absent" | "compact";
  draggableHandle?: boolean;
  onStartDragOut?: () => void;
  className?: string;
}

export function TapeLabel({
  stemId,
  size = "default",
  draggableHandle = false,
  onStartDragOut,
  className,
}: TapeLabelProps) {
  const name = stemName(stemId);
  const sizeClass = size === "absent" ? "tape--absent" : size === "compact" ? "tape--compact" : "";
  const classes = [
    "tape",
    `stem-${stemId}`,
    sizeClass,
    draggableHandle ? "tape--draggable" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  if (!draggableHandle) {
    return <span className={classes}>{name}</span>;
  }

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    onStartDragOut?.();
  };

  return (
    <div className="tape-wrap">
      <div
        className={classes}
        draggable
        tabIndex={0}
        role="button"
        aria-label={`${name}. Drag into your DAW or a folder`}
        onDragStart={handleDragStart}
      >
        {name}
      </div>
      <span className="tape__tip" role="tooltip">
        Drag into your DAW or a folder
      </span>
    </div>
  );
}
