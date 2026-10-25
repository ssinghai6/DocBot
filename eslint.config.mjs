import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Non-source trees (DOCBOT-2302): Python venvs ship vendored JS that
    // crashes the linter; agent worktrees/tooling dirs duplicate src/.
    ".venv/**",
    "venv/**",
    ".claude/**",
    ".opencode/**",
    ".vercel/**",
    "coverage/**",
    "finetune/**",
    "DocBot_Real_Test/**",
  ]),
]);

export default eslintConfig;
