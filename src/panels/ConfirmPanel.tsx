import { DialogPanel } from "../components/primitives/DialogPanel";

interface ConfirmPanelProps {
  title: string;
  body: string;
  confirmLabel: string;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ConfirmPanel({
  title,
  body,
  confirmLabel,
  onCancel,
  onConfirm,
}: ConfirmPanelProps) {
  return (
    <DialogPanel
      title={title}
      size="narrow"
      onClose={onCancel}
      footer={
        <>
          <button type="button" className="console-key" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="console-key console-key--danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </>
      }
    >
      <p>{body}</p>
    </DialogPanel>
  );
}
