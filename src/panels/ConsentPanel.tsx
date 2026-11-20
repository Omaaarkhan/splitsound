import type { ConsentRequest } from "../stores/uiStore";
import { formatBytes } from "../utils/format";
import { DialogPanel } from "../components/primitives/DialogPanel";
import "./ConsentPanel.css";

interface ConsentPanelProps {
  request: ConsentRequest;
  onCancel: () => void;
  onAccept: () => void;
  onSplitWithout?: () => void;
}

export function ConsentPanel({ request, onCancel, onAccept, onSplitWithout }: ConsentPanelProps) {
  const sizeText = formatBytes(request.model.downloadBytes);
  const termsText =
    request.model.terms ??
    "This model has additional licence terms. Review them before downloading.";

  return (
    <DialogPanel
      title="Guitar and piano need one more model"
      size="wide"
      onClose={onCancel}
      footer={
        <div className="consent-actions">
          <button type="button" className="console-key" onClick={onCancel}>
            Cancel
          </button>
          {request.kind === "split" && onSplitWithout ? (
            <button type="button" className="console-key" onClick={onSplitWithout}>
              Split without guitar and piano
            </button>
          ) : null}
          <button type="button" className="primary-btn" onClick={onAccept}>
            {request.kind === "split"
              ? `Download from ${request.model.source} and split`
              : `Download from ${request.model.source}`}
          </button>
        </div>
      }
    >
      <div className="consent-body">
        <p>{termsText}</p>
        <p className="consent-body__size">It is a {sizeText} download.</p>
      </div>
    </DialogPanel>
  );
}
