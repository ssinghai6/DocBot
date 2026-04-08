"use client"

import React, { useState } from "react"
import {
  FileText, Search, ChevronDown, Loader2,
  AlertCircle, Download, CheckCircle2,
} from "lucide-react"

type CompanyResult = {
  cik: string
  ticker: string
  name: string
}

type Filing = {
  accession_number: string
  filing_type: string
  filing_date: string
  primary_document: string
  description: string
}

type IngestResult = {
  session_id?: string
  chunks_created?: number
  cached?: boolean
  filing_date?: string
  accession_number?: string
  error?: string
}

interface EdgarPanelProps {
  onFilingIngested: (sessionId: string, label: string) => void
  showToast: (type: "success" | "error" | "warning" | "info", message: string) => void
}

const FILING_TYPES = ["10-K", "10-Q", "8-K", "20-F", "6-K"]

export default function EdgarPanel({ onFilingIngested, showToast }: EdgarPanelProps) {
  const [expanded, setExpanded] = useState(false)

  // Search state
  const [searchQuery, setSearchQuery] = useState("")
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<CompanyResult[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)

  // Selected company state
  const [selectedCompany, setSelectedCompany] = useState<CompanyResult | null>(null)
  const [filingType, setFilingType] = useState("10-K")
  const [filingCount, setFilingCount] = useState(3)

  // Filings list state
  const [filings, setFilings] = useState<Filing[]>([])
  const [loadingFilings, setLoadingFilings] = useState(false)

  // Ingest state
  const [ingesting, setIngesting] = useState<string | null>(null) // accession_number being ingested
  const [batchIngesting, setBatchIngesting] = useState(false)
  const [ingestResults, setIngestResults] = useState<Record<string, IngestResult>>({})

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setSearchError(null)
    setSearchResults([])
    setSelectedCompany(null)
    setFilings([])
    setIngestResults({})

    try {
      const res = await fetch("/api/edgar/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery.trim() }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Search failed")
      }
      const data = await res.json()
      setSearchResults(data.results ?? [])
      if ((data.results ?? []).length === 0) {
        setSearchError("No companies found")
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Search failed"
      setSearchError(msg)
    } finally {
      setSearching(false)
    }
  }

  const handleSelectCompany = async (company: CompanyResult) => {
    setSelectedCompany(company)
    setFilings([])
    setIngestResults({})
    setLoadingFilings(true)

    try {
      const res = await fetch("/api/edgar/filings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cik: company.cik, filing_type: filingType, count: filingCount }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Failed to list filings")
      }
      const data = await res.json()
      setFilings(data.filings ?? [])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to list filings"
      showToast("error", msg)
    } finally {
      setLoadingFilings(false)
    }
  }

  const handleRefreshFilings = async () => {
    if (!selectedCompany) return
    setLoadingFilings(true)
    setIngestResults({})
    try {
      const res = await fetch("/api/edgar/filings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cik: selectedCompany.cik, filing_type: filingType, count: filingCount }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Failed to list filings")
      }
      const data = await res.json()
      setFilings(data.filings ?? [])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to refresh filings"
      showToast("error", msg)
    } finally {
      setLoadingFilings(false)
    }
  }

  const handleIngestSingle = async (filing: Filing) => {
    if (!selectedCompany) return
    setIngesting(filing.accession_number)

    try {
      const res = await fetch("/api/edgar/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cik: selectedCompany.cik,
          accession_number: filing.accession_number,
          primary_document: filing.primary_document,
          filing_type: filing.filing_type,
          filing_date: filing.filing_date,
          company_name: selectedCompany.name,
          ticker: selectedCompany.ticker,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Ingest failed")
      }
      const result: IngestResult = await res.json()
      setIngestResults(prev => ({ ...prev, [filing.accession_number]: result }))

      if (result.session_id) {
        const label = `${selectedCompany.ticker} ${filing.filing_type} (${filing.filing_date})`
        onFilingIngested(result.session_id, label)
        showToast("success", result.cached ? `${label} loaded from cache` : `${label} ingested (${result.chunks_created} chunks)`)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Ingest failed"
      setIngestResults(prev => ({ ...prev, [filing.accession_number]: { error: msg } }))
      showToast("error", msg)
    } finally {
      setIngesting(null)
    }
  }

  const handleBatchIngest = async () => {
    if (!selectedCompany) return
    setBatchIngesting(true)

    try {
      const res = await fetch("/api/edgar/ingest-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cik: selectedCompany.cik,
          ticker: selectedCompany.ticker,
          company_name: selectedCompany.name,
          filing_type: filingType,
          count: filingCount,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Batch ingest failed")
      }
      const data = await res.json()
      const results: IngestResult[] = data.results ?? []

      // Update ingest results map
      const newResults: Record<string, IngestResult> = {}
      for (const r of results) {
        const key = r.accession_number ?? r.filing_date ?? ""
        if (key) newResults[key] = r
      }
      setIngestResults(prev => ({ ...prev, ...newResults }))

      // Notify for each successful ingest
      const ingested = results.filter(r => r.session_id)
      if (ingested.length > 0) {
        const last = ingested[ingested.length - 1]
        if (last.session_id) {
          onFilingIngested(last.session_id, `${selectedCompany.ticker} ${filingType} (${filingCount} filings)`)
        }
        showToast("success", `Ingested ${data.summary?.ingested ?? ingested.length} of ${data.summary?.total ?? results.length} filings`)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Batch ingest failed"
      showToast("error", msg)
    } finally {
      setBatchIngesting(false)
    }
  }

  return (
    <div className="mb-5">
      <h3 className="text-sm font-semibold mb-3 text-[var(--color-text-primary)] flex items-center gap-2">
        <FileText className="w-4 h-4 text-[var(--color-success-500)]" />
        SEC Filings
      </h3>

      {/* Expand/collapse toggle */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)]/50 hover:border-[var(--color-success-500)]/40 hover:bg-[var(--color-success-500)]/5 transition-all text-xs text-[var(--color-text-secondary)]"
      >
        <Search className="w-3.5 h-3.5 text-[var(--color-success-500)] shrink-0" />
        <span>Search SEC EDGAR</span>
        <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>

      {expanded && (
        <div className="space-y-3 pt-3">
          {/* Search input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleSearch() }}
              placeholder="Ticker or company name"
              className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs placeholder-[var(--color-text-quaternary)] focus:outline-none focus:border-[var(--color-success-500)]/40"
            />
            <button
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              className="px-3 py-2 rounded-lg bg-[var(--color-success-500)]/20 hover:bg-[var(--color-success-500)]/30 border border-[var(--color-success-500)]/30 text-[var(--color-success-500)] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {searching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            </button>
          </div>

          {searchError && (
            <p className="text-[10px] text-red-400 px-1 flex items-center gap-1">
              <AlertCircle className="w-3 h-3 shrink-0" />
              {searchError}
            </p>
          )}

          {/* Search results */}
          {searchResults.length > 0 && !selectedCompany && (
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {searchResults.map(company => (
                <button
                  key={company.cik}
                  onClick={() => handleSelectCompany(company)}
                  className="w-full text-left px-3 py-2 rounded-lg hover:bg-[var(--color-success-500)]/10 border border-transparent hover:border-[var(--color-success-500)]/20 transition-all"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-[var(--color-success-500)]">{company.ticker}</span>
                    <span className="text-xs text-[var(--color-text-secondary)] truncate flex-1">{company.name}</span>
                  </div>
                  <p className="text-[10px] text-[var(--color-text-quaternary)] mt-0.5">CIK: {company.cik}</p>
                </button>
              ))}
            </div>
          )}

          {/* Selected company + filing controls */}
          {selectedCompany && (
            <div className="space-y-2">
              {/* Selected company header */}
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-success-500)]/10 border border-[var(--color-success-500)]/20">
                <FileText className="w-3.5 h-3.5 text-[var(--color-success-500)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <span className="text-xs font-medium text-gray-300">{selectedCompany.ticker}</span>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] truncate">{selectedCompany.name}</p>
                </div>
                <button
                  onClick={() => {
                    setSelectedCompany(null)
                    setFilings([])
                    setIngestResults({})
                  }}
                  className="text-[var(--color-text-tertiary)] hover:text-gray-300 text-[10px]"
                >
                  Change
                </button>
              </div>

              {/* Filing type + count */}
              <div className="flex gap-2">
                <select
                  value={filingType}
                  onChange={e => setFilingType(e.target.value)}
                  className="flex-1 px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs focus:outline-none focus:border-[var(--color-success-500)]/40"
                >
                  {FILING_TYPES.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
                <select
                  value={filingCount}
                  onChange={e => setFilingCount(Number(e.target.value))}
                  className="w-16 px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs focus:outline-none focus:border-[var(--color-success-500)]/40"
                >
                  {[1, 2, 3, 5, 10].map(n => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
                <button
                  onClick={handleRefreshFilings}
                  disabled={loadingFilings}
                  className="px-2 py-1.5 rounded-lg bg-[var(--color-success-500)]/20 hover:bg-[var(--color-success-500)]/30 border border-[var(--color-success-500)]/30 text-[var(--color-success-500)] text-xs transition-all disabled:opacity-50"
                  title="Refresh filings"
                >
                  {loadingFilings ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                </button>
              </div>

              {/* Batch ingest button */}
              <button
                onClick={handleBatchIngest}
                disabled={batchIngesting || !filings.length}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-success-500)]/20 hover:bg-[var(--color-success-500)]/30 border border-[var(--color-success-500)]/30 text-[var(--color-success-500)] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {batchIngesting ? (
                  <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Ingesting filings...</>
                ) : (
                  <><Download className="w-3.5 h-3.5" /> Ingest All {filingCount} {filingType}s</>
                )}
              </button>

              {/* Filing list */}
              {loadingFilings && (
                <div className="flex items-center justify-center py-4 text-xs text-[var(--color-text-tertiary)]">
                  <Loader2 className="w-4 h-4 animate-spin mr-2" /> Loading filings...
                </div>
              )}

              {filings.length > 0 && (
                <div className="space-y-1.5 max-h-48 overflow-y-auto">
                  {filings.map(filing => {
                    const result = ingestResults[filing.accession_number]
                    const isIngesting = ingesting === filing.accession_number
                    const isIngested = result?.session_id != null
                    const hasError = result?.error != null

                    return (
                      <div
                        key={filing.accession_number}
                        className={`px-3 py-2 rounded-lg border transition-all ${
                          isIngested
                            ? "bg-[var(--color-success-500)]/10 border-[var(--color-success-500)]/20"
                            : hasError
                            ? "bg-red-500/10 border-red-500/20"
                            : "bg-[var(--color-bg-elevated)]/50 border-[var(--color-border-subtle)] hover:border-[var(--color-success-500)]/20"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-medium text-gray-300">{filing.filing_type}</span>
                              <span className="text-[10px] text-[var(--color-text-tertiary)]">{filing.filing_date}</span>
                            </div>
                            {filing.description && (
                              <p className="text-[10px] text-[var(--color-text-quaternary)] truncate mt-0.5">{filing.description}</p>
                            )}
                          </div>

                          {isIngested ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-[var(--color-success-500)] shrink-0" />
                          ) : hasError ? (
                            <AlertCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                          ) : (
                            <button
                              onClick={() => handleIngestSingle(filing)}
                              disabled={isIngesting || batchIngesting}
                              className="text-[var(--color-success-500)] hover:text-[var(--color-success-500)]/80 disabled:opacity-50 shrink-0"
                              title="Ingest this filing"
                            >
                              {isIngesting ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Download className="w-3.5 h-3.5" />
                              )}
                            </button>
                          )}
                        </div>

                        {isIngested && result.cached && (
                          <p className="text-[10px] text-[var(--color-success-500)] mt-1">Loaded from cache</p>
                        )}
                        {isIngested && !result.cached && (
                          <p className="text-[10px] text-[var(--color-success-500)] mt-1">{result.chunks_created} chunks created</p>
                        )}
                        {hasError && (
                          <p className="text-[10px] text-red-400 mt-1">{result.error}</p>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
