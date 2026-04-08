"use client"

import React from "react"
import { Upload, FileText, Loader2, CheckCircle2, AlertCircle } from "lucide-react"
import type { FileUploadState } from "./types"

interface FileUploadZoneProps {
  fileUploadState: FileUploadState
  uploadProgress: number | null
  uploadedFiles: File[]
  deepVisualMode: boolean
  onDeepVisualModeChange: (value: boolean) => void
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>
  // Unified upload: CSV/SQLite routing
  onDbUpload?: (file: File, type: "csv" | "sqlite") => void
  dbUploadState?: "idle" | "uploading" | "connected" | "error"
}

function getFileUploadBorderColor(fileUploadState: FileUploadState): string {
  switch (fileUploadState) {
    case "dragover":  return "border-[var(--color-success-500)] bg-[var(--color-success-500)]/5"
    case "uploading": return "border-[var(--color-cyan-500)] bg-[var(--color-cyan-500)]/5"
    case "success":   return "border-[var(--color-success-500)] bg-[var(--color-success-500)]/5"
    case "error":     return "border-[#ef4444] bg-[var(--color-danger-500)]/5"
    default:          return "border-[var(--color-cyan-500)]/30 bg-[var(--color-bg-surface)]/50"
  }
}

/** Route a file to the correct handler based on extension */
function getFileType(name: string): "pdf" | "csv" | "sqlite" | null {
  const ext = name.split(".").pop()?.toLowerCase()
  if (ext === "pdf") return "pdf"
  if (ext === "csv") return "csv"
  if (ext === "sqlite" || ext === "db" || ext === "sqlite3") return "sqlite"
  return null
}

export default function FileUploadZone({
  fileUploadState,
  uploadProgress,
  uploadedFiles,
  deepVisualMode,
  onDeepVisualModeChange,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileChange,
  fileInputRef,
  onDbUpload,
  dbUploadState,
}: FileUploadZoneProps) {
  /** Split a FileList into PDF and DB (CSV/SQLite) buckets */
  const splitFiles = (files: FileList | File[]) => {
    const pdfs: File[] = []
    const dbs: { file: File; type: "csv" | "sqlite" }[] = []
    for (let i = 0; i < files.length; i++) {
      const f = (files as FileList)[i] ?? (files as File[])[i]
      const ftype = getFileType(f.name)
      if (ftype === "pdf") pdfs.push(f)
      else if (ftype === "csv" || ftype === "sqlite") dbs.push({ file: f, type: ftype })
    }
    return { pdfs, dbs }
  }

  /** Build a synthetic input change event whose files contain only `pdfs` */
  const dispatchPdfs = (pdfs: File[]) => {
    if (pdfs.length === 0) return
    const dt = new DataTransfer()
    pdfs.forEach((f) => dt.items.add(f))
    const synthetic = { target: { files: dt.files } } as unknown as React.ChangeEvent<HTMLInputElement>
    onFileChange(synthetic)
  }

  /** Handle unified file input change — routes PDFs and CSV/SQLite in parallel */
  const handleUnifiedFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const { pdfs, dbs } = splitFiles(files)

    if (dbs.length > 0 && onDbUpload) {
      dbs.forEach(({ file, type }) => onDbUpload(file, type))
    }
    if (pdfs.length > 0) {
      dispatchPdfs(pdfs)
    }

    e.target.value = ""
  }

  /** Handle unified drop — routes PDFs and CSV/SQLite in parallel */
  const handleUnifiedDrop = (e: React.DragEvent) => {
    const files = e.dataTransfer?.files
    if (!files || files.length === 0) {
      onDrop(e)
      return
    }

    const { pdfs, dbs } = splitFiles(files)

    if (pdfs.length === 0 && dbs.length === 0) {
      onDrop(e)
      return
    }

    e.preventDefault()
    e.stopPropagation()

    if (dbs.length > 0 && onDbUpload) {
      dbs.forEach(({ file, type }) => onDbUpload(file, type))
    }
    if (pdfs.length > 0) {
      dispatchPdfs(pdfs)
    }
  }

  const isUploading = fileUploadState === "uploading" || dbUploadState === "uploading"

  return (
    <div className="mb-5">
      <h3 className="text-sm font-semibold mb-3 text-[var(--color-text-primary)] flex items-center gap-2">
        <Upload className="w-4 h-4 text-[var(--color-cyan-500)]" />
        Upload
        <span className="ml-auto text-[10px] text-[var(--color-text-tertiary)] font-normal">PDF, CSV, SQLite</span>
      </h3>

      {/* Deep Visual Mode Toggle */}
      <div className="mb-4 bg-[var(--color-bg-elevated)]/50 rounded-xl p-3 border border-[var(--color-border-subtle)]">
        <label className="flex items-center justify-between cursor-pointer group">
          <div className="flex items-center gap-3">
            <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepVisualMode ? "bg-[var(--color-cyan-500)]" : ""}`}>
              <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepVisualMode ? "translate-x-4" : ""}`} />
            </div>
            <div>
              <span className="text-sm font-medium text-gray-300 group-hover:text-[var(--color-text-primary)] transition-colors">Deep Visual</span>
              <p className="text-[10px] text-[var(--color-text-tertiary)]">Full page OCR analysis</p>
            </div>
          </div>
          <input
            type="checkbox"
            className="hidden"
            checked={deepVisualMode}
            onChange={() => onDeepVisualModeChange(!deepVisualMode)}
          />
        </label>
      </div>

      {/* Upload Area */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={handleUnifiedDrop}
        className={`
          group relative border-2 border-dashed rounded-2xl p-5 text-center cursor-pointer transition-all
          ${getFileUploadBorderColor(fileUploadState)}
          hover:border-[var(--color-cyan-500)]/60 hover:shadow-[0_0_30px_rgba(102,126,234,0.15)]
          ${isUploading ? "pointer-events-none" : ""}
        `}
      >
        {fileUploadState === "uploading" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 relative">
              <div className="absolute inset-0 rounded-2xl border-2 border-[var(--color-cyan-500)]/30" />
              <div className="absolute inset-0 rounded-2xl border-2 border-transparent border-t-[var(--color-cyan-500)] animate-spin" />
              <Loader2 className="absolute inset-0 m-auto w-6 h-6 text-[var(--color-cyan-500)] animate-spin" />
            </div>
            <p className="text-sm font-medium text-gray-300">Processing documents...</p>
            {uploadProgress !== null && (
              <div className="mt-2 w-full bg-[#ffffff10] rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[var(--color-cyan-500)] to-[var(--color-cyan-600)] transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        ) : fileUploadState === "success" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[var(--color-success-500)]/20 flex items-center justify-center">
              <CheckCircle2 className="w-6 h-6 text-[var(--color-success-500)]" />
            </div>
            <p className="text-sm font-medium text-[var(--color-success-500)]">Upload complete!</p>
          </div>
        ) : fileUploadState === "error" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[var(--color-danger-500)]/20 flex items-center justify-center">
              <AlertCircle className="w-6 h-6 text-[#ef4444]" />
            </div>
            <p className="text-sm font-medium text-[#ef4444]">Upload failed</p>
          </div>
        ) : fileUploadState === "dragover" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[var(--color-success-500)]/20 flex items-center justify-center animate-bounce">
              <FileText className="w-6 h-6 text-[var(--color-success-500)]" />
            </div>
            <p className="text-sm font-medium text-[var(--color-success-500)]">Drop files to upload</p>
          </div>
        ) : (
          <div className="relative">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-[var(--color-cyan-500)]/20 to-[var(--color-cyan-600)]/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Upload className="w-6 h-6 text-[var(--color-cyan-500)]" />
            </div>
            <p className="text-sm font-medium text-gray-300 group-hover:text-[var(--color-text-primary)]">Drop files here</p>
            <p className="text-xs text-[var(--color-text-tertiary)] mt-1">PDF, CSV, or SQLite • click to browse</p>
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="application/pdf,.csv,.sqlite,.db,.sqlite3"
          multiple
          onChange={handleUnifiedFileChange}
        />
      </div>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div className="mt-3 space-y-2">
          {uploadedFiles.map((f, i) => (
            <div key={i} className="flex items-center text-xs bg-[var(--color-success-500)]/10 px-3 py-2.5 rounded-xl border border-[var(--color-success-500)]/20">
              <FileText className="w-4 h-4 mr-2 text-[var(--color-success-500)] shrink-0" />
              <span className="truncate flex-1 text-gray-300">{f.name}</span>
              <span className="text-[10px] text-[var(--color-text-tertiary)] ml-2">
                {(f.size / 1024).toFixed(1)}KB
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
