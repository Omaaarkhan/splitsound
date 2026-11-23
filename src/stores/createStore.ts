import { useRef, useSyncExternalStore } from "react";

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

export function shallowEqual<A, B>(a: A, b: B): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || a === null || typeof b !== "object" || b === null) {
    return false;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (!Object.is(a[i], b[i])) return false;
    }
    return true;
  }
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  const objA = a as Record<string, unknown>;
  const objB = b as Record<string, unknown>;
  for (const key of keysA) {
    if (!Object.prototype.hasOwnProperty.call(objB, key) || !Object.is(objA[key], objB[key])) {
      return false;
    }
  }
  return true;
}

export function useStore<T, S = T>(store: Store<T>, selector?: (state: T) => S): S {
  return useSyncExternalStore(
    store.subscribe,
    () => (selector ? selector(store.getState()) : (store.getState() as unknown as S)),
    () => (selector ? selector(store.getState()) : (store.getState() as unknown as S)),
  );
}

export function useShallowStore<T, S>(store: Store<T>, selector: (state: T) => S): S {
  const cacheRef = useRef<{ hasValue: boolean; value: S }>({
    hasValue: false,
    value: undefined as S,
  });
  const getSnapshot = () => {
    const next = selector(store.getState());
    if (cacheRef.current.hasValue && shallowEqual(cacheRef.current.value, next)) {
      return cacheRef.current.value;
    }
    cacheRef.current = { hasValue: true, value: next };
    return next;
  };
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}
