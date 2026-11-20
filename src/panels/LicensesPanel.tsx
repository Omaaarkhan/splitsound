import { useStore } from "../stores/createStore";
import { modelsStore } from "../stores/modelsStore";
import { DialogPanel } from "../components/primitives/DialogPanel";
import "./LicensesPanel.css";

interface LicensesPanelProps {
  onClose: () => void;
}

export function LicensesPanel({ onClose }: LicensesPanelProps) {
  const { models } = useStore(modelsStore);

  return (
    <DialogPanel
      title="Licences"
      onClose={onClose}
      footer={
        <button type="button" className="console-key" onClick={onClose}>
          Close
        </button>
      }
    >
      <div className="licenses">
        <div className="licenses__item">
          <div className="licenses__name">SplitSound</div>
          <p className="licenses__meta">MIT licence.</p>
        </div>

        <div className="licenses__item">
          <div className="licenses__name">Fonts</div>
          <p className="licenses__meta">
            Archivo (SIL Open Font Licence 1.1) and Just Another Hand (Apache Licence 2.0), bundled
            locally with the app.
          </p>
        </div>

        {models.map((m) => (
          <div key={m.id} className="licenses__item">
            <div className="licenses__name">
              {m.name} — {m.license}
            </div>
            <p className="licenses__meta">
              {m.attribution}, from {m.source}
            </p>
            {m.terms ? <p className="licenses__meta">{m.terms}</p> : null}
          </div>
        ))}
      </div>
    </DialogPanel>
  );
}
