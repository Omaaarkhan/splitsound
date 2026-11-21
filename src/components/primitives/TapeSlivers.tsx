import type { StemId } from "../../ipc/contract";
import { stemName } from "../../utils/stems";
import "./TapeSlivers.css";

interface TapeSliversProps {
  stemIds: readonly StemId[];
  labelPrefix?: string;
}

export function TapeSlivers({ stemIds, labelPrefix = "Stems" }: TapeSliversProps) {
  const names = stemIds.map(stemName).join(", ");
  return (
    <span className="slivers">
      <span className="sr-only">
        {labelPrefix}: {names}
      </span>
      {stemIds.map((id) => (
        <span key={id} className={`sliver stem-${id}`} aria-hidden="true" />
      ))}
    </span>
  );
}
