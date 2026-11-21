import type { ModelDownloadProgress, ModelStatus } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { getScanModels } from "../utils/format";
import { createStore } from "./createStore";

export interface ModelsState {
  models: ModelStatus[];
  progressById: Record<string, ModelDownloadProgress>;
  /** Models belonging to the active scan-model download batch so sequential progress stays monotonic. */
  scanBatchIds: string[];
  loaded: boolean;
}

const initialModelsState = (): ModelsState => ({
  models: [],
  progressById: {},
  scanBatchIds: [],
  loaded: false,
});

export const modelsStore = createStore<ModelsState>(initialModelsState);

/** Models whose download was explicitly triggered from Settings or "Split song". */
const userTriggeredIds = new Set<string>();
/** Models that have emitted download progress during the current attempt. */
const progressedIds = new Set<string>();
/** Models cancelled by the user so their final models:changed does not surface as a failure. */
const cancelledIds = new Set<string>();

export function resetModelDownloadTracking(): void {
  userTriggeredIds.clear();
  progressedIds.clear();
  cancelledIds.clear();
}

export function markUserTriggeredDownloads(modelIds: readonly string[]): void {
  for (const id of modelIds) {
    cancelledIds.delete(id);
    userTriggeredIds.add(id);
  }
}

export function markUserCancelledDownloads(modelIds?: readonly string[]): void {
  const targets =
    modelIds ??
    modelsStore
      .getState()
      .models.filter((m) => m.downloadProgress !== null || userTriggeredIds.has(m.id))
      .map((m) => m.id);
  for (const id of targets) {
    cancelledIds.add(id);
    userTriggeredIds.delete(id);
    progressedIds.delete(id);
  }
}

export async function refreshModels(): Promise<ModelStatus[]> {
  const models = await invoke("list_models");
  modelsStore.setState((prev) => {
    const anyDownloading = models.some((m) => m.downloadProgress !== null);
    const scanBatchIds = anyDownloading
      ? getScanModels(models)
          .filter((m) => !m.installed)
          .map((m) => m.id)
      : [];
    return {
      ...prev,
      models,
      scanBatchIds: scanBatchIds.length > 0 ? scanBatchIds : prev.scanBatchIds,
      loaded: true,
    };
  });
  return models;
}

export function applyModelsChanged(models: ModelStatus[]): { failedModelIds: string[] } {
  const prevState = modelsStore.getState();
  const prevById = new Map(prevState.models.map((m) => [m.id, m]));
  const failedModelIds: string[] = [];

  for (const next of models) {
    const prev = prevById.get(next.id);
    const wasDownloading =
      prev?.downloadProgress !== null ||
      progressedIds.has(next.id) ||
      next.id in prevState.progressById;

    if (next.installed) {
      userTriggeredIds.delete(next.id);
      progressedIds.delete(next.id);
      cancelledIds.delete(next.id);
    } else if (next.downloadProgress === null) {
      if (cancelledIds.has(next.id)) {
        cancelledIds.delete(next.id);
      } else if (userTriggeredIds.has(next.id) && (wasDownloading || !prev?.installed)) {
        failedModelIds.push(next.id);
      }
      userTriggeredIds.delete(next.id);
      progressedIds.delete(next.id);
    }
  }

  modelsStore.setState((prev) => {
    const progressById: Record<string, ModelDownloadProgress> = {};
    for (const m of models) {
      if (m.downloadProgress !== null && prev.progressById[m.id]) {
        progressById[m.id] = prev.progressById[m.id];
      }
    }
    const stillDownloading = models.some((m) => m.downloadProgress !== null);
    return {
      models,
      progressById,
      scanBatchIds: stillDownloading ? prev.scanBatchIds : [],
      loaded: true,
    };
  });

  return { failedModelIds };
}

export function applyModelProgress(progress: ModelDownloadProgress): void {
  cancelledIds.delete(progress.modelId);
  progressedIds.add(progress.modelId);
  modelsStore.setState((prev) => {
    const scanIds = new Set(getScanModels(prev.models).map((m) => m.id));
    let nextBatch = prev.scanBatchIds;
    if (nextBatch.length === 0 && scanIds.has(progress.modelId)) {
      nextBatch = getScanModels(prev.models)
        .filter((m) => !m.installed || m.id === progress.modelId)
        .map((m) => m.id);
    }
    return {
      ...prev,
      scanBatchIds: nextBatch,
      progressById: {
        ...prev.progressById,
        [progress.modelId]: progress,
      },
      models: prev.models.map((m) =>
        m.id === progress.modelId
          ? {
              ...m,
              downloadProgress:
                progress.totalBytes > 0 ? progress.receivedBytes / progress.totalBytes : 0,
            }
          : m,
      ),
    };
  });
}

export async function downloadModels(
  modelIds: string[],
  acceptRestricted?: boolean,
): Promise<void> {
  await invoke("download_models", { modelIds, acceptRestricted });
}

export async function cancelModelDownloads(): Promise<void> {
  markUserCancelledDownloads();
  await invoke("cancel_model_downloads");
}

export async function removeModel(modelId: string): Promise<void> {
  userTriggeredIds.delete(modelId);
  progressedIds.delete(modelId);
  cancelledIds.delete(modelId);
  await invoke("remove_model", { modelId });
}

export function findUninstalledRestrictedModel(): ModelStatus | undefined {
  return modelsStore.getState().models.find((m) => m.restricted && !m.installed);
}
