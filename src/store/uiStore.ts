import { create } from "zustand";
import { persist } from "zustand/middleware";

export type InspectorTab = "query" | "schema" | "profile" | "metadata" | "artifact";

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

      inspectorTab: "query",
      setInspectorTab: (tab) => set({ inspectorTab: tab }),

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
              : artifact.type === "chart"
              ? "artifact"
              : artifact.type === "table"
              ? "artifact"
              : artifact.type === "citations"
              ? "metadata"
              : s.inspectorTab
            : s.inspectorTab,
        })),

      commandPaletteOpen: false,
      setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
      toggleCommandPalette: () => set((s) => ({ commandPaletteOpen: !s.commandPaletteOpen })),
    }),
    {
      name: "docbot-ui-store",
      partialize: (s) => ({
        sidebarCollapsed: s.sidebarCollapsed,
        inspectorOpen: s.inspectorOpen,
        inspectorTab: s.inspectorTab,
      }),
    }
  )
);
