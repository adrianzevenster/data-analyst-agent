import { useState, useRef, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  BarChart3,
  Upload,
  ChevronDown,
  ChevronRight,
  Database,
  Copy,
  Check,
  X,
  Download,
  Trash2,
} from 'lucide-react'
import clsx from 'clsx'
import { getDatasets, getSample, uploadFile, getModels, deleteModel, getLLMHealth, getRagEval } from '../lib/api'
import type { Dataset } from '../types/api'

interface SidebarProps {
  datasetId: string | null
  onDatasetChange: (id: string) => void
  conversationId: string | null
}

function SectionHeader({ label }: { label: string }) {
  return (
    <p className="text-slate-400 uppercase text-xs tracking-wider font-semibold mb-2">
      {label}
    </p>
  )
}

function DataPreview({ datasetId }: { datasetId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['sample', datasetId],
    queryFn: () => getSample(datasetId, 5),
    staleTime: 60_000,
  })

  if (isLoading) return <p className="text-slate-400 text-xs mt-1">Loading…</p>
  const rows = data?.data ?? []
  if (!rows.length) return <p className="text-slate-400 text-xs mt-1">No rows.</p>
  const cols = Object.keys(rows[0])

  return (
    <div className="mt-2 overflow-auto max-h-36 thin-scroll rounded border border-slate-700">
      <table className="text-xs w-full">
        <thead>
          <tr className="bg-slate-800">
            {cols.map((c) => (
              <th key={c} className="text-left px-2 py-1 text-slate-300 whitespace-nowrap font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-slate-800/40' : 'bg-slate-800/20'}>
              {cols.map((c) => (
                <td key={c} className="px-2 py-1 text-slate-300 whitespace-nowrap max-w-[100px] truncate">
                  {String(row[c] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DatasetItem({
  ds,
  active,
  onSelect,
}: {
  ds: Dataset
  active: boolean
  onSelect: () => void
}) {
  const [showPreview, setShowPreview] = useState(false)

  return (
    <div
      className={clsx(
        'rounded-lg p-2 cursor-pointer transition-colors',
        active ? 'bg-indigo-600/20 ring-1 ring-indigo-500' : 'hover:bg-slate-800'
      )}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-slate-100 text-sm font-medium truncate">{ds.filename}</p>
          <p className="text-slate-400 text-xs">
            {ds.n_rows.toLocaleString()} rows × {ds.n_cols} cols
          </p>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            setShowPreview((v) => !v)
          }}
          className="ml-2 text-slate-400 hover:text-slate-200 transition-colors"
          title="Toggle preview"
        >
          {showPreview ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>
      {showPreview && <DataPreview datasetId={ds.dataset_id} />}
    </div>
  )
}

function ModelRegistry() {
  const [open, setOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const qc = useQueryClient()
  const { data: models = [] } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    refetchInterval: open ? 15_000 : false,
  })

  function copyId(id: string) {
    navigator.clipboard.writeText(id)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  function downloadModel(id: string, modelType: string, targetCol: string) {
    const a = document.createElement('a')
    a.href = `/api/models/${id}/download`
    a.download = `${modelType}__${targetCol}__${id.slice(0, 8)}.joblib`
    a.click()
  }

  async function handleDelete(id: string) {
    setDeletingId(id)
    try {
      await deleteModel(id)
      await qc.invalidateQueries({ queryKey: ['models'] })
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-slate-400 uppercase text-xs tracking-wider font-semibold mb-2 hover:text-slate-300 transition-colors w-full"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Model Registry
      </button>
      {open && (
        <div>
          {models.length === 0 ? (
            <p className="text-slate-500 text-xs">No models trained yet.</p>
          ) : (
            <div className="space-y-1.5">
              {models.map((m) => (
                <div
                  key={m.model_id}
                  className="bg-slate-800 rounded px-2 py-2"
                >
                  <div className="flex items-start justify-between gap-1">
                    <div className="min-w-0 flex-1">
                      <p className="text-slate-200 text-xs font-mono">{m.model_id.slice(0, 8)}…</p>
                      <p className="text-slate-400 text-xs mt-0.5">
                        {m.model_type} · {m.task_type}
                      </p>
                      <p className="text-slate-400 text-xs">
                        target: <span className="text-indigo-400">{m.target_col}</span>
                        {m.log_transform_target && (
                          <span className="ml-1 text-blue-400" title="log1p transform was applied to target">·log</span>
                        )}
                      </p>
                      <p className="text-slate-500 text-xs">{m.feature_cols.length} features · {m.created_at.slice(0, 10)}</p>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <button
                        onClick={() => copyId(m.model_id)}
                        className="text-slate-400 hover:text-slate-200 transition-colors p-0.5"
                        title="Copy model ID"
                      >
                        {copiedId === m.model_id ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                      </button>
                      <button
                        onClick={() => downloadModel(m.model_id, m.model_type, m.target_col)}
                        className="text-slate-400 hover:text-slate-200 transition-colors p-0.5"
                        title="Download model artifact (.joblib)"
                      >
                        <Download size={12} />
                      </button>
                      <button
                        onClick={() => handleDelete(m.model_id)}
                        disabled={deletingId === m.model_id}
                        className="text-slate-500 hover:text-red-400 transition-colors p-0.5 disabled:opacity-40"
                        title="Delete model"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function LLMHealth() {
  const [open, setOpen] = useState(false)
  const { data: stats } = useQuery({
    queryKey: ['llm-health'],
    queryFn: getLLMHealth,
    refetchInterval: open ? 30_000 : false,
  })

  const hasData = stats && stats.window_size > 0

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-slate-400 uppercase text-xs tracking-wider font-semibold mb-2 hover:text-slate-300 transition-colors w-full"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        LLM Health
      </button>
      {open && (
        <div>
          {!hasData ? (
            <p className="text-slate-500 text-xs">No LLM calls recorded yet.</p>
          ) : (
            <div className="space-y-1">
              <div className="grid grid-cols-2 gap-1">
                <div className="bg-slate-800 rounded px-2 py-1.5">
                  <p className="text-slate-400 text-xs">Avg latency</p>
                  <p className="text-slate-100 text-sm font-medium">{stats.avg_latency_ms.toFixed(0)} ms</p>
                </div>
                <div className="bg-slate-800 rounded px-2 py-1.5">
                  <p className="text-slate-400 text-xs">Error rate</p>
                  <p className="text-slate-100 text-sm font-medium">{(stats.error_rate * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="bg-slate-800 rounded px-2 py-1.5">
                <p className="text-slate-400 text-xs">Tokens sampled</p>
                <p className="text-slate-100 text-sm font-medium">{stats.total_tokens_sampled.toLocaleString()}</p>
              </div>
              <p className="text-slate-500 text-xs mt-1">Last {stats.window_size} calls</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const TOOL_GROUPS: { label: string; tools: { name: string; description: string }[] }[] = [
  {
    label: 'Explore',
    tools: [
      { name: 'profile_dataset',      description: 'Column stats, types, missingness.' },
      { name: 'data_quality_report',  description: 'Missing %, skewness, percentiles.' },
      { name: 'auto_insights',        description: 'Ranked findings across quality, relationships, anomalies and trends.' },
      { name: 'correlation_analysis', description: 'Strongest numeric and categorical associations.' },
      { name: 'trend_analysis',       description: 'Trend and period-over-period change on a datetime column.' },
    ],
  },
  {
    label: 'Query',
    tools: [
      { name: 'duckdb_query',    description: "SQL over the active dataset (table alias 't')." },
      { name: 'multidim_pivot',  description: 'Pivot / multi-dim aggregation.' },
    ],
  },
  {
    label: 'Quality & Detection',
    tools: [
      { name: 'missingness_matrix',         description: 'Columns with highest missing ratios.' },
      { name: 'overrepresented_categories', description: 'Dominant values in a categorical column.' },
      { name: 'skewed_features',            description: 'Numeric features with high skewness.' },
      { name: 'anomaly_scan',               description: 'Outlier detection via IsolationForest.' },
      { name: 'kmeans_clusters',            description: 'KMeans clustering on numeric columns.' },
    ],
  },
  {
    label: 'ML',
    tools: [
      { name: 'train_supervised_model',   description: 'Train classification or regression model on a target column.' },
      { name: 'score_with_model',         description: 'Apply a trained model to the current dataset.' },
      { name: 'explain_model',            description: 'SHAP / permutation feature importance for a stored model.' },
      { name: 'evaluate_ml_predictions',  description: 'Evaluate prediction output columns (classification, regression, forecast).' },
    ],
  },
  {
    label: 'Charts',
    tools: [
      { name: 'simple_bar_spec', description: 'Bar chart from x / y columns.' },
      { name: 'histogram_spec',  description: 'Distribution histogram for a numeric column.' },
      { name: 'line_spec',       description: 'Line chart of y over x.' },
      { name: 'scatter_spec',    description: 'Scatter plot of y vs x with correlation.' },
    ],
  },
]

function AvailableTools() {
  const [open, setOpen] = useState(false)

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-slate-400 uppercase text-xs tracking-wider font-semibold mb-2 hover:text-slate-300 transition-colors w-full"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Available Tools
      </button>
      {open && (
        <div className="space-y-3">
          {TOOL_GROUPS.map((group) => (
            <div key={group.label}>
              <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide mb-1">
                {group.label}
              </p>
              <div className="space-y-1">
                {group.tools.map((t) => (
                  <div key={t.name} className="bg-slate-800 rounded px-2 py-1.5">
                    <p className="text-slate-200 text-xs font-mono">{t.name}</p>
                    <p className="text-slate-500 text-xs mt-0.5 leading-tight">{t.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function RagEval() {
  const [open, setOpen] = useState(false)
  const { data } = useQuery({
    queryKey: ['rag-eval'],
    queryFn: getRagEval,
    staleTime: 5 * 60_000,
  })

  if (!data?.available) return null

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-slate-400 uppercase text-xs tracking-wider font-semibold mb-2 hover:text-slate-300 transition-colors w-full"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        RAG Retrieval Eval
      </button>
      {open && (
        <div className="space-y-1">
          <p className="text-slate-500 text-xs mb-2">{data.n_queries} labeled queries</p>
          <div className="grid grid-cols-2 gap-1">
            {Object.entries(data.aggregate)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([k, stats]) => (
                <div key={k} className="bg-slate-800 rounded px-2 py-1.5">
                  <p className="text-slate-400 text-xs">Recall@{k}</p>
                  <p className="text-slate-100 text-sm font-medium">
                    {(stats.recall_at_k * 100).toFixed(0)}%
                  </p>
                  <p className="text-slate-400 text-xs">Prec@{k}</p>
                  <p className="text-slate-100 text-sm font-medium">
                    {(stats.precision_at_k * 100).toFixed(0)}%
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Sidebar({ datasetId, onDatasetChange, conversationId }: SidebarProps) {
  const qc = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets,
    refetchInterval: 30_000,
  })

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) setPendingFile(file)
  }, [])

  async function handleUpload() {
    if (!pendingFile) return
    setUploading(true)
    setUploadError(null)
    setUploadSuccess(false)
    try {
      const result = await uploadFile(pendingFile)
      onDatasetChange(result.dataset_id)
      setUploadSuccess(true)
      setPendingFile(null)
      await qc.invalidateQueries({ queryKey: ['datasets'] })
      setTimeout(() => setUploadSuccess(false), 3000)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed'
      setUploadError(msg)
    } finally {
      setUploading(false)
    }
  }

  const activeDatasets = datasets.length
  const otherDatasets = datasets.filter((d) => d.dataset_id !== datasetId)

  return (
    <aside className="w-64 flex-shrink-0 bg-slate-900 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-4 border-b border-slate-700">
        <BarChart3 size={20} className="text-indigo-400 flex-shrink-0" />
        <span className="text-slate-100 font-semibold text-sm">Data Analyst</span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto thin-scroll px-3 py-3 space-y-5">
        {/* Upload */}
        <div>
          <SectionHeader label="Upload Data" />
          <div
            className={clsx(
              'border-2 border-dashed rounded-lg p-3 text-center transition-colors cursor-pointer',
              dragOver
                ? 'border-indigo-500 bg-indigo-500/10'
                : 'border-slate-700 hover:border-slate-500'
            )}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={16} className="text-slate-400 mx-auto mb-1" />
            {pendingFile ? (
              <p className="text-slate-300 text-xs font-medium truncate px-1">{pendingFile.name}</p>
            ) : (
              <p className="text-slate-500 text-xs">Drop file or click</p>
            )}
            <p className="text-slate-600 text-xs mt-0.5">CSV · XLSX · PDF · Image</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.pdf,.png,.jpg,.jpeg,.webp"
            onChange={(e) => e.target.files?.[0] && setPendingFile(e.target.files[0])}
          />
          {pendingFile && (
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-xs font-medium py-1.5 rounded-lg transition-colors"
              >
                {uploading ? 'Uploading…' : 'Upload'}
              </button>
              <button
                onClick={() => setPendingFile(null)}
                className="text-slate-400 hover:text-slate-200 transition-colors"
              >
                <X size={14} />
              </button>
            </div>
          )}
          {uploadSuccess && (
            <p className="text-green-400 text-xs mt-1 flex items-center gap-1">
              <Check size={12} /> Uploaded successfully
            </p>
          )}
          {uploadError && <p className="text-red-400 text-xs mt-1">{uploadError}</p>}
        </div>

        {/* Datasets */}
        <div>
          <SectionHeader label="Datasets" />
          {activeDatasets === 0 ? (
            <p className="text-slate-500 text-xs">No datasets uploaded yet.</p>
          ) : (
            <div className="space-y-1">
              {datasets.map((ds) => (
                <DatasetItem
                  key={ds.dataset_id}
                  ds={ds}
                  active={ds.dataset_id === datasetId}
                  onSelect={() => onDatasetChange(ds.dataset_id)}
                />
              ))}
            </div>
          )}
          {activeDatasets > 1 && (
            <p className="text-slate-500 text-xs mt-2">
              SQL: <span className="text-slate-300 font-mono">t</span> (active)
              {otherDatasets.slice(0, 2).map((d) => (
                <span key={d.dataset_id}>
                  {' '}+{' '}
                  <span className="text-slate-300 font-mono">
                    {d.filename.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 16)}
                  </span>
                </span>
              ))}
            </p>
          )}
        </div>

        <div className="border-t border-slate-700" />

        {/* Model Registry */}
        <ModelRegistry />

        <div className="border-t border-slate-700" />

        {/* LLM Health */}
        <LLMHealth />

        <div className="border-t border-slate-700" />

        {/* RAG Eval */}
        <RagEval />

        <div className="border-t border-slate-700" />

        {/* Available Tools */}
        <AvailableTools />
      </div>

      {/* Footer — conversation ID */}
      {conversationId && (
        <div className="px-3 py-2 border-t border-slate-700">
          <p className="text-slate-500 text-xs font-mono truncate" title={conversationId}>
            <Database size={10} className="inline mr-1" />
            {conversationId.slice(0, 8)}…
          </p>
        </div>
      )}
    </aside>
  )
}
