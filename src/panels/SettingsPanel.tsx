import { useMemo, useState } from "react";
import type { ModelStatus, Quality, Settings, UpdateInfo } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { pickAnyFiles } from "../ipc/platform";
import { useStore } from "../stores/createStore";
import { checkForUpdatesManually, startInstallUpdate } from "../stores/index";
import { cancelModelDownloads, modelsStore, removeModel } from "../stores/modelsStore";
import { settingsStore, updateSettings } from "../stores/settingsStore";
import { showNotice } from "../stores/uiStore";
import { formatIpcError } from "../utils/errors";
import { formatBytes, formatImportedModelsNotice, formatModelMeta } from "../utils/format";
import { Checkbox } from "../components/primitives/Checkbox";
import { DialogPanel } from "../components/primitives/DialogPanel";
import { SelectMenu, type SelectOption } from "../components/primitives/SelectMenu";
import { tip } from "../components/primitives/Tooltip";
import "./SettingsPanel.css";

interface SettingsPanelProps {
  onClose: () => void;
  onDownloadModel: (model: ModelStatus) => void;
  onRequestClearLibrary: () => void;
  onOpenLicenses: () => void;
}

const ACCEL_OPTIONS: readonly { value: Settings["acceleration"]; label: string }[] = [
  { value: "auto", label: "Automatic" },
  { value: "cpu", label: "Never" },
  { value: "gpu", label: "Always" },
];

export function SettingsPanel({
  onClose,
  onDownloadModel,
  onRequestClearLibrary,
  onOpenLicenses,
}: SettingsPanelProps) {
  const { settings, appInfo } = useStore(settingsStore);
  const { models } = useStore(modelsStore);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [updateResult, setUpdateResult] = useState<
    { status: "up-to-date" } | { status: "available"; update: UpdateInfo } | null
  >(null);

  const threadOptions = useMemo<SelectOption<number>[]>(() => {
    const opts: SelectOption<number>[] = [{ value: 0, label: "Automatic" }];
    const max = Math.max(2, appInfo.cpuThreads);
    const candidates = [1, 2, 4, 6, 8, 12, 16];
    for (const c of candidates) {
      if (c <= max) {
        opts.push({ value: c, label: `${c} ${c === 1 ? "thread" : "threads"}` });
      }
    }
    if (!opts.some((o) => o.value === max)) {
      opts.push({ value: max, label: `${max} threads` });
    }
    return opts;
  }, [appInfo.cpuThreads]);

  const handleCheckUpdates = async () => {
    if (checkingUpdate) return;
    setCheckingUpdate(true);
    const res = await checkForUpdatesManually();
    if (res.status === "error") {
      setUpdateResult(null);
    } else {
      setUpdateResult(res);
    }
    setCheckingUpdate(false);
  };

  const handleAddModelFiles = async () => {
    const paths = await pickAnyFiles();
    if (paths.length === 0) return;
    try {
      const recognised = await invoke("import_model_files", { paths });
      showNotice(formatImportedModelsNotice(recognised.length), {
        tone: recognised.length === 0 ? "error" : "info",
      });
    } catch (err) {
      showNotice(formatIpcError(err), { tone: "error" });
    }
  };

  return (
    <DialogPanel
      title="Settings"
      size="wide"
      onClose={onClose}
      headerAction={
        <button type="button" className="console-key console-key--compact" onClick={onClose}>
          Close
        </button>
      }
    >
      <div className="settings">
        <section className="settings-section" aria-label="Splitting">
          <h3 className="settings-section__title">Splitting</h3>

          <div className="settings-row">
            <div className="settings-row__label">
              <span className="settings-row__name">Default quality</span>
            </div>
            <div className="settings-switch" role="radiogroup" aria-label="Default quality">
              {(["fast", "best"] as Quality[]).map((q) => {
                const active = settings.defaultQuality === q;
                return (
                  <button
                    key={q}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    className={
                      active
                        ? "console-key console-key--compact console-key--selected"
                        : "console-key console-key--compact"
                    }
                    onClick={() => void updateSettings({ defaultQuality: q })}
                    {...tip(q === "fast" ? "Fast quality" : "Best quality", {
                      detail:
                        q === "fast"
                          ? "Uses the smaller model for quicker splits."
                          : "Uses the larger model for cleaner separation.",
                      placement: "top",
                    })}
                  >
                    {q === "fast" ? "Fast" : "Best"}
                  </button>
                );
              })}
            </div>
          </div>

          {appInfo.gpu.available ? (
            <div className="settings-row">
              <div
                className="settings-row__label"
                {...tip("Hardware acceleration", {
                  detail:
                    "Automatic picks the fastest backend for your computer. Choose Never to force CPU.",
                  placement: "top",
                })}
              >
                <span className="settings-row__name">Use the graphics processor</span>
                {appInfo.gpu.name ? (
                  <span className="settings-row__sub">{appInfo.gpu.name}</span>
                ) : null}
              </div>
              <div
                className="settings-switch"
                role="radiogroup"
                aria-label="Use the graphics processor"
              >
                {ACCEL_OPTIONS.map((opt) => {
                  const active = settings.acceleration === opt.value;
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      className={
                        active
                          ? "console-key console-key--compact console-key--selected"
                          : "console-key console-key--compact"
                      }
                      onClick={() => void updateSettings({ acceleration: opt.value })}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}

          <div className="settings-row">
            <div
              className="settings-row__label"
              {...tip("CPU threads", {
                detail:
                  "How many processor cores splitting can use. Lower this if your computer feels sluggish while splitting.",
                placement: "top",
              })}
            >
              <span className="settings-row__name">Processor threads</span>
            </div>
            <SelectMenu
              label="Processor threads"
              value={settings.threads}
              options={threadOptions}
              onChange={(threads) => void updateSettings({ threads })}
            />
          </div>
        </section>

        <section className="settings-section" aria-label="Models">
          <h3 className="settings-section__title">Models</h3>
          <ul className="model-list">
            {models.map((model) => {
              const isDownloading = model.downloadProgress !== null;
              const pct = isDownloading ? Math.round((model.downloadProgress ?? 0) * 100) : 0;
              return (
                <li key={model.id} className="model-row">
                  <div className="model-row__info">
                    <div className="model-row__head">
                      <span className="model-row__name">{model.name}</span>
                      <span className="model-row__meta">{formatModelMeta(model)}</span>
                    </div>
                    <p className="model-row__desc">{model.description}</p>
                    {model.restricted && model.terms ? (
                      <p className="model-row__terms">{model.terms}</p>
                    ) : null}
                  </div>

                  <div className="model-row__action">
                    {isDownloading ? (
                      <div className="model-row__prog">
                        <span className="model-row__pct">{pct}%</span>
                        <button
                          type="button"
                          className="console-key console-key--compact"
                          onClick={() => void cancelModelDownloads()}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : model.installed ? (
                      <button
                        type="button"
                        className="console-key console-key--compact"
                        onClick={() => void removeModel(model.id)}
                      >
                        Remove
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="console-key console-key--compact"
                        onClick={() => onDownloadModel(model)}
                      >
                        Download
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
          <div className="models-footer">
            <p className="models-footer__help">
              For computers that can&apos;t download: choose model files you fetched yourself.
            </p>
            <button
              type="button"
              className="console-key console-key--compact"
              onClick={() => void handleAddModelFiles()}
              {...tip("Import models offline", {
                detail:
                  "Install verified model files you downloaded manually. Every file is checked by SHA-256.",
                placement: "top",
              })}
            >
              Add model files
            </button>
          </div>
        </section>

        <section className="settings-section" aria-label="Library">
          <h3 className="settings-section__title">Library</h3>
          <div className="settings-row">
            <div className="settings-row__label">
              <span className="settings-row__name">{formatBytes(appInfo.libraryBytes)} used</span>
              <span className="settings-path">{appInfo.libraryDir}</span>
            </div>
            <button
              type="button"
              className="console-key console-key--compact"
              onClick={onRequestClearLibrary}
              {...tip("Clear all saved songs", {
                detail:
                  "Removes all separated stems stored in SplitSound. Your original audio files are untouched.",
                placement: "left",
              })}
            >
              Clear library
            </button>
          </div>
        </section>

        <section className="settings-section" aria-label="Updates">
          <h3 className="settings-section__title">Updates</h3>
          <div className="settings-row">
            <div className="settings-row__label">
              <div className="settings-updates-meta">
                <span className="settings-row__name">Version {appInfo.version}</span>
                <Checkbox
                  checked={settings.checkForUpdates}
                  onChange={(checkForUpdates) => void updateSettings({ checkForUpdates })}
                >
                  Check automatically
                </Checkbox>
              </div>
              {updateResult ? (
                <span className="settings-row__sub">
                  {updateResult.status === "up-to-date"
                    ? "SplitSound is up to date."
                    : `SplitSound ${updateResult.update.version} is available.`}
                </span>
              ) : null}
            </div>
            <div className="settings-inline-actions">
              {updateResult?.status === "available" ? (
                <button
                  type="button"
                  className="console-key console-key--compact console-key--lit"
                  onClick={() => {
                    onClose();
                    void startInstallUpdate();
                  }}
                >
                  Update and restart
                </button>
              ) : null}
              <button
                type="button"
                className="console-key console-key--compact"
                disabled={checkingUpdate}
                onClick={() => void handleCheckUpdates()}
              >
                {checkingUpdate ? "Checking" : "Check for updates"}
              </button>
            </div>
          </div>
        </section>

        <section className="settings-section" aria-label="About">
          <h3 className="settings-section__title">About</h3>
          <div className="settings-row">
            <p className="settings-row__sub">
              SplitSound never uploads your music. Everything happens on this computer.
            </p>
            <button type="button" className="settings-link" onClick={onOpenLicenses}>
              Licences
            </button>
          </div>
        </section>
      </div>
    </DialogPanel>
  );
}
