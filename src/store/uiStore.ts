import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Lineage } from "@/components/types";

export const INSPECTOR_TABS = ["lineage", "sources", "query", "data", "run"] as const;
export type InspectorTab = (typeof INSPECTOR_TABS)[number];

/** Persisted values from older builds ("metadata", "artifact") fall back to lineage. */
export function normalizeInspectorTab(tab: string): InspectorTab {
  return (INSPECTOR_TABS as readonly string[]).includes(tab) ? (tab as InspectorTab) : "lineage";
}

export interface SelectedArtifact {
  messageId: string;
  type: "chart" | "sql" | "table" | "code" | "citations";
  /** Message-local payload the inspector needs to render (sql text, chart b64, etc.) */
  payload?: Record<string, unknown>;
}

interface UIState {
  // Panel visibility / layout
  sidebarCollapsed: boolean;
  inspectorOpen: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setInspectorOpen: (open: boolean) => void;
  toggleInspector: () => void;

  // Inspector tab state
  inspectorTab: InspectorTab;
  setInspectorTab: (tab: InspectorTab) => void;

  // Selected artifact (syncs message cards → inspector)
  selectedArtifact: SelectedArtifact | null;
  selectArtifact: (artifact: SelectedArtifact | null) => void;

  // DOCBOT-1510: lineage of the answer being inspected (latest answer by default)
  selectedLineage: Lineage | null;
  /** Source id (LineageSource.id) to highlight in the Sources tab */
  focusSourceId: string | null;
  selectLineage: (lineage: Lineage | null, opts?: { tab?: InspectorTab; focusSourceId?: string | null }) => void;

  // Command palette
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
  toggleCommandPalette: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      inspectorOpen: true,
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setInspectorOpen: (open) => set({ inspectorOpen: open }),
      toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),

      inspectorTab: "lineage",
      setInspectorTab: (tab) => set({ inspectorTab: tab }),

      selectedLineage: null,
      focusSourceId: null,
      selectLineage: (lineage, opts) =>
        set((s) => ({
          selectedLineage: lineage,
          focusSourceId: opts?.focusSourceId ?? null,
          inspectorTab: opts?.tab ?? s.inspectorTab,
        })),

      selectedArtifact: null,
      selectArtifact: (artifact) =>
        set((s) => ({
          selectedArtifact: artifact,
          // auto-open inspector when artifact is selected
          inspectorOpen: artifact ? true : s.inspectorOpen,
          // jump to sensible default tab per artifact type
          inspectorTab: artifact
            ? artifact.type === "sql"
              ? "query"
              : artifact.type === "chart" || artifact.type === "table" || artifact.type === "code"
              ? "data"
              : artifact.type === "citations"
              ? "sources"
              : s.inspectorTab
            : s.inspectorTab,
        })),

      commandPaletteOpen: false,
      setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
      toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),
    }),
    {
      name: "docbot-ui-store",
      version: 2,
      migrate: (persisted) => {
        const p = (persisted ?? {}) as Partial<UIState>;
        return { ...p, inspectorTab: normalizeInspectorTab(p.inspectorTab ?? "") } as UIState;
      },
      partialize: (s) => ({
        sidebarCollapsed: s.sidebarCollapsed,
        inspectorOpen: s.inspectorOpen,
        inspectorTab: s.inspectorTab,
      }),
    }
  )
);
