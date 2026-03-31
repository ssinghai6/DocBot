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
    case "dragover":  return "border-[#10b981] bg-[#10b981]/5"
    case "uploading": return "border-[#667eea] bg-[#667eea]/5"
    case "success":   return "border-[#10b981] bg-[#10b981]/5"
    case "error":     return "border-[#ef4444] bg-[#ef4444]/5"
    default:          return "border-[#667eea]/30 bg-[#12121a]/50"
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
  /** Handle unified file input change — routes PDF vs CSV/SQLite */
  const handleUnifiedFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    // Check if any file is CSV/SQLite and route to DB handler
    const file = files[0]
    const ftype = getFileType(file.name)

    if (ftype === "csv" && onDbUpload) {
      onDbUpload(file, "csv")
      e.target.value = ""
    } else if (ftype === "sqlite" && onDbUpload) {
      onDbUpload(file, "sqlite")
      e.target.value = ""
    } else {
      // PDF — use existing handler
      onFileChange(e)
    }
  }

  /** Handle unified drop — routes PDF vs CSV/SQLite */
  const handleUnifiedDrop = (e: React.DragEvent) => {
    const files = e.dataTransfer?.files
    if (files && files.length > 0) {
      const file = files[0]
      const ftype = getFileType(file.name)

      if ((ftype === "csv" || ftype === "sqlite") && onDbUpload) {
        e.preventDefault()
        e.stopPropagation()
        onDbUpload(file, ftype)
        return
      }
    }
    // PDF or fallback — use existing drop handler
    onDrop(e)
  }

  const isUploading = fileUploadState === "uploading" || dbUploadState === "uploading"

  return (
    <div className="mb-5">
      <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
        <Upload className="w-4 h-4 text-[#667eea]" />
        Upload
        <span className="ml-auto text-[10px] text-gray-500 font-normal">PDF, CSV, SQLite</span>
      </h3>

      {/* Deep Visual Mode Toggle */}
      <div className="mb-4 bg-[#1a1a24]/50 rounded-xl p-3 border border-[#ffffff06]">
        <label className="flex items-center justify-between cursor-pointer group">
          <div className="flex items-center gap-3">
            <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepVisualMode ? "bg-[#667eea]" : ""}`}>
              <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepVisualMode ? "translate-x-4" : ""}`} />
            </div>
            <div>
              <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">Deep Visual</span>
              <p className="text-[10px] text-gray-500">Full page OCR analysis</p>
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
          hover:border-[#667eea]/60 hover:shadow-[0_0_30px_rgba(102,126,234,0.15)]
          ${isUploading ? "pointer-events-none" : ""}
        `}
      >
        {fileUploadState === "uploading" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 relative">
              <div className="absolute inset-0 rounded-2xl border-2 border-[#667eea]/30" />
              <div className="absolute inset-0 rounded-2xl border-2 border-transparent border-t-[#667eea] animate-spin" />
              <Loader2 className="absolute inset-0 m-auto w-6 h-6 text-[#667eea] animate-spin" />
            </div>
            <p className="text-sm font-medium text-gray-300">Processing documents...</p>
            {uploadProgress !== null && (
              <div className="mt-2 w-full bg-[#ffffff10] rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#667eea] to-[#764ba2] transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        ) : fileUploadState === "success" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#10b981]/20 flex items-center justify-center">
              <CheckCircle2 className="w-6 h-6 text-[#10b981]" />
            </div>
            <p className="text-sm font-medium text-[#10b981]">Upload complete!</p>
          </div>
        ) : fileUploadState === "error" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#ef4444]/20 flex items-center justify-center">
              <AlertCircle className="w-6 h-6 text-[#ef4444]" />
            </div>
            <p className="text-sm font-medium text-[#ef4444]">Upload failed</p>
          </div>
        ) : fileUploadState === "dragover" ? (
          <div className="py-2">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#10b981]/20 flex items-center justify-center animate-bounce">
              <FileText className="w-6 h-6 text-[#10b981]" />
            </div>
            <p className="text-sm font-medium text-[#10b981]">Drop files to upload</p>
          </div>
        ) : (
          <div className="relative">
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-[#667eea]/20 to-[#764ba2]/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Upload className="w-6 h-6 text-[#667eea]" />
            </div>
            <p className="text-sm font-medium text-gray-300 group-hover:text-white">Drop files here</p>
            <p className="text-xs text-gray-500 mt-1">PDF, CSV, or SQLite • click to browse</p>
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
            <div key={i} className="flex items-center text-xs bg-[#10b981]/10 px-3 py-2.5 rounded-xl border border-[#10b981]/20">
              <FileText className="w-4 h-4 mr-2 text-[#10b981] shrink-0" />
              <span className="truncate flex-1 text-gray-300">{f.name}</span>
              <span className="text-[10px] text-gray-500 ml-2">
                {(f.size / 1024).toFixed(1)}KB
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
