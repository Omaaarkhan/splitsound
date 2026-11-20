import { useSyncExternalStore } from "react";

export interface Store<T> {
  getState: () => T;
  setState: (updater: T | ((prev: T) => T)) => void;
  subscribe: (listener: () => void) => () => void;
  reset: () => void;
}

export function createStore<T>(initialState: () => T): Store<T> {
  let state = initialState();
  const listeners = new Set<() => void>();

  const getState = () => state;

  const setState = (updater: T | ((prev: T) => T)) => {
    const next = typeof updater === "function" ? (updater as (prev: T) => T)(state) : updater;
    if (Object.is(next, state)) return;
    state = next;
    for (const listener of listeners) {
      listener();
    }
  };

  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  };

  const reset = () => {
    state = initialState();
    for (const listener of listeners) {
      listener();
    }
  };

  return { getState, setState, subscribe, reset };
}

export function useStore<T, S = T>(store: Store<T>, selector?: (state: T) => S): S {
  return useSyncExternalStore(
    store.subscribe,
    () => (selector ? selector(store.getState()) : (store.getState() as unknown as S)),
    () => (selector ? selector(store.getState()) : (store.getState() as unknown as S)),
  );
}
