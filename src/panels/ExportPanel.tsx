import { useMemo, useState } from "react";
import type { ExportFormat, Session, StemId } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { pickDirectory } from "../ipc/platform";
import { useStore } from "../stores/createStore";
import { settingsStore, updateSettings } from "../stores/settingsStore";
import { showNotice } from "../stores/uiStore";
import { formatIpcError } from "../utils/errors";
import { EXPORT_FORMATS, sortStemsForDisplay } from "../utils/stems";
import { Checkbox } from "../components/primitives/Checkbox";
import { DialogPanel } from "../components/primitives/DialogPanel";
import { SelectMenu } from "../components/primitives/SelectMenu";
import { TapeLabel } from "../components/primitives/TapeLabel";
import { tip } from "../components/primitives/Tooltip";
import "./ExportPanel.css";

interface ExportPanelProps {
  session: Session;
  onClose: () => void;
}

export function ExportPanel({ session, onClose }: ExportPanelProps) {
  const { settings } = useStore(settingsStore);
  const orderedStems = useMemo(() => sortStemsForDisplay(session.stems), [session.stems]);

  const [selected, setSelected] = useState<Set<StemId>>(
    () => new Set(orderedStems.filter((s) => s.present).map((s) => s.id)),
  );
  const [format, setFormat] = useState<ExportFormat>(settings.exportFormat);
  const [includeMixdown, setIncludeMixdown] = useState(false);
  const [destinationDir, setDestinationDir] = useState<string>(
    settings.exportDir ?? "/Users/demo/Music/SplitSound",
  );
  const [exporting, setExporting] = useState(false);

  const fileCount = selected.size + (includeMixdown ? 1 : 0);

  const toggleStem = (id: StemId, checked: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const handleChangeDir = async () => {
    const picked = await pickDirectory(destinationDir);
    if (picked) {
      setDestinationDir(picked);
    }
  };

  const handleExport = async () => {
    if (fileCount === 0 || exporting) return;
    setExporting(true);
    try {
      const stemIds = orderedStems.filter((s) => selected.has(s.id)).map((s) => s.id);
      const result = await invoke("export_stems", {
        sessionId: session.id,
        options: {
          stemIds,
          format,
          destinationDir,
          includeMixdown,
        },
      });
      await updateSettings({ exportFormat: format, exportDir: destinationDir });
      onClose();
      const n = result.files.length;
      showNotice(`Exported ${n} ${n === 1 ? "file" : "files"}`, {
        actionLabel: "Show in folder",
        onAction: () => {
          void invoke("reveal_path", { path: result.destinationDir });
        },
      });
    } catch (err) {
      setExporting(false);
      showNotice(formatIpcError(err), { tone: "error" });
    }
  };

  return (
    <DialogPanel
      title="Export stems"
      onClose={onClose}
      footer={
        <>
          <button type="button" className="console-key" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary-btn"
            disabled={fileCount === 0 || exporting}
            onClick={() => void handleExport()}
          >
            {exporting ? "Exporting" : `Export ${fileCount} ${fileCount === 1 ? "file" : "files"}`}
          </button>
        </>
      }
    >
      <div className="export-form">
        <div className="export-group">
          <span className="export-group__label">Stems</span>
          <ul className="export-stems">
            {orderedStems.map((stem) => {
              const checked = selected.has(stem.id);
              return (
                <li key={stem.id} className="export-stem-row">
                  <Checkbox checked={checked} onChange={(next) => toggleStem(stem.id, next)}>
                    <TapeLabel stemId={stem.id} size={stem.present ? "compact" : "absent"} />
                  </Checkbox>
                  {!stem.present ? (
                    <span className="export-stem-row__absent">None heard in this song</span>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </div>

        <div className="export-row">
          <span
            className="export-group__label"
            {...tip("Audio format", {
              detail: "WAV and FLAC are lossless. MP3 is smaller and easier to share.",
              placement: "top",
            })}
          >
            Format
          </span>
          <SelectMenu
            label="Export format"
            value={format}
            options={EXPORT_FORMATS}
            onChange={setFormat}
          />
        </div>

        <Checkbox
          checked={includeMixdown}
          onChange={setIncludeMixdown}
          tooltip={tip("Custom mixdown", {
            detail:
              "Saves one extra combined file using your current solo, mute, and volume faders.",
            placement: "top",
          })}
        >
          Also export one file with these stems mixed at the current levels
        </Checkbox>

        <div className="export-group">
          <span className="export-group__label">Destination folder</span>
          <div className="export-folder">
            <div
              className="export-folder__path"
              {...tip("Export folder", { detail: destinationDir, placement: "top" })}
            >
              {destinationDir}
            </div>
            <button type="button" className="console-key" onClick={() => void handleChangeDir()}>
              Change
            </button>
          </div>
        </div>
      </div>
    </DialogPanel>
  );
}
