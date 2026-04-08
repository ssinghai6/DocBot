"use client";

import { useEffect, useState } from "react";

export type Breakpoint = "xs" | "sm" | "md" | "lg" | "xl";

/**
 * Responsive breakpoint hook based on CSS media queries.
 *
 *   xs  <  640px   — single column, drawer sidebar
 *   sm  640–899    — rail sidebar, no inspector
 *   md  900–1279   — full sidebar, inspector auto-collapsed
 *   lg  1280–1535  — all three panels
 *   xl  ≥ 1536     — all three panels, wider defaults
 */
export function useBreakpoint(): Breakpoint {
  const [bp, setBp] = useState<Breakpoint>("lg");

  useEffect(() => {
    const queries: Array<[Breakpoint, MediaQueryList]> = [
      ["xl", window.matchMedia("(min-width: 1536px)")],
      ["lg", window.matchMedia("(min-width: 1280px)")],
      ["md", window.matchMedia("(min-width: 900px)")],
      ["sm", window.matchMedia("(min-width: 640px)")],
    ];

    const resolve = () => {
      for (const [name, mql] of queries) {
        if (mql.matches) {
          setBp(name);
          return;
        }
      }
      setBp("xs");
    };

    resolve();
    queries.forEach(([, mql]) => mql.addEventListener("change", resolve));
    return () => queries.forEach(([, mql]) => mql.removeEventListener("change", resolve));
  }, []);

  return bp;
}

export function useIsDesktop(): boolean {
  const bp = useBreakpoint();
  return bp === "lg" || bp === "xl";
}
