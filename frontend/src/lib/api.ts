import axios from 'axios'
import type {
  Dataset,
  ConversationTurn,
  Experiment,
  Model,
  LLMStats,
  RagEvalResponse,
  TrainingJob,
  JudgeStats,
  JudgeHistoryResponse,
} from '../types/api'

const client = axios.create({ baseURL: '/api' })

export async function getDatasets(): Promise<Dataset[]> {
  const { data } = await client.get<Dataset[]>('/datasets')
  return data
}

export async function getSample(
  datasetId: string,
  limit = 50
): Promise<{ data: Record<string, unknown>[] }> {
  const { data } = await client.get(`/datasets/${datasetId}/sample`, {
    params: { limit },
  })
  return data
}

export async function uploadFile(file: File): Promise<{
  dataset_id: string
  filename: string
  n_rows: number
  n_cols: number
  notes: string[]
}> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await client.post('/uploads', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getHistory(
  conversationId: string
): Promise<{ conversation_id: string; turns: ConversationTurn[] }> {
  const { data } = await client.get(`/chat/${conversationId}/history`)
  return data
}

export async function getModels(): Promise<Model[]> {
  const { data } = await client.get<Model[]>('/models')
  return data
}

export async function deleteModel(modelId: string): Promise<void> {
  await client.delete(`/models/${modelId}`)
}

export async function getLLMHealth(): Promise<LLMStats> {
  const { data } = await client.get<LLMStats>('/health/llm')
  return data
}

export async function getRagEval(): Promise<RagEvalResponse> {
  const { data } = await client.get<RagEvalResponse>('/health/rag-eval')
  return data
}

export async function runRagEval(): Promise<RagEvalResponse> {
  const { data } = await client.post<RagEvalResponse>('/health/rag-eval/run')
  return data
}

export async function scoreFile(modelId: string, file: File): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await client.post<Blob>(`/models/${modelId}/score-file`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  })
  return data
}

export async function getExperiments(params?: {
  dataset_id?: string
  target_col?: string
  limit?: number
}): Promise<Experiment[]> {
  const { data } = await client.get<Experiment[]>('/experiments', { params })
  return data
}

export async function startTrainingJob(params: {
  dataset_id: string
  target_col: string
  model_type?: string
  tune?: boolean
  cv_folds?: number
}): Promise<{ job_id: string; status: string }> {
  const { data } = await client.post('/training/jobs', params)
  return data
}

export async function getTrainingJob(jobId: string): Promise<TrainingJob> {
  const { data } = await client.get<TrainingJob>(`/training/jobs/${jobId}`)
  return data
}

export async function listTrainingJobs(limit = 20): Promise<TrainingJob[]> {
  const { data } = await client.get<TrainingJob[]>('/training/jobs', { params: { limit } })
  return data
}

export async function getJudgeStats(): Promise<JudgeStats> {
  const { data } = await client.get<JudgeStats>('/health/llm-judge')
  return data
}

export async function getJudgeHistory(limit = 100): Promise<JudgeHistoryResponse> {
  const { data } = await client.get<JudgeHistoryResponse>('/health/llm-judge/history', { params: { limit } })
  return data
}

export interface CorpusFile {
  filename: string
  size_bytes: number
  modified_at: number
}

export interface CorpusStatus {
  files: CorpusFile[]
  ingest_running: boolean
  last_chunks_indexed: number | null
  last_ingest_error: string | null
}

export async function listCorpusFiles(): Promise<CorpusStatus> {
  const { data } = await client.get<CorpusStatus>('/corpus')
  return data
}

export async function uploadCorpusFile(file: File): Promise<{ filename: string; status: string }> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await client.post<{ filename: string; status: string }>('/corpus/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function deleteCorpusFile(filename: string): Promise<{ status: string }> {
  const { data } = await client.delete<{ status: string }>(`/corpus/files/${filename}`)
  return data
}

export interface QualityTrendDay {
  day: string
  avg_score: number
  n: number
  min_score: number
  max_score: number
}

export async function getQualityTrend(days = 30): Promise<{ days: number; data: QualityTrendDay[] }> {
  const { data } = await client.get('/health/quality-trend', { params: { days } })
  return data
}

export async function triggerEvalRun(params?: { n?: number; max_age_days?: number }): Promise<{
  run_id: string; status: string
}> {
  const { data } = await client.post('/eval/run', null, { params })
  return data
}

export async function pollEvalRunStatus(runId: string): Promise<{
  run_id: string; status: string; n_sampled?: number; n_judged?: number; n_failed?: number; avg_score?: number | null; error?: string
}> {
  const { data } = await client.get(`/eval/run/status/${runId}`)
  return data
}

export async function connectPostgres(connectionString: string, query: string, datasetName?: string): Promise<import('../types/api').UploadResponse> {
  const { data } = await client.post('/connectors/postgres', {
    connection_string: connectionString,
    query,
    dataset_name: datasetName || null,
  })
  return data
}

export async function connectSqlite(file: File, query: string, datasetName?: string): Promise<import('../types/api').UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('query', query)
  if (datasetName) form.append('dataset_name', datasetName)
  const { data } = await client.post('/connectors/sqlite', form)
  return data
}

export async function connectUrl(url: string, format: 'auto' | 'csv' | 'parquet' | 'json' = 'auto', datasetName?: string): Promise<import('../types/api').UploadResponse> {
  const { data } = await client.post('/connectors/url', {
    url,
    format,
    dataset_name: datasetName || null,
  })
  return data
}

export async function generateReport(conversationId: string, useLlm = true): Promise<{
  report: string; format: string; source: string; n_findings: number
}> {
  const { data } = await client.post('/reports/generate', {
    conversation_id: conversationId,
    use_llm: useLlm,
  })
  return data
}

export interface DatasetAnnotation {
  description: string
  columns: Record<string, string>
}

export async function getAnnotations(datasetId: string): Promise<DatasetAnnotation & { dataset_id: string }> {
  const { data } = await client.get(`/annotations/${datasetId}`)
  return data
}

export async function saveAnnotations(
  datasetId: string,
  ann: DatasetAnnotation & { dataset_filename?: string },
): Promise<DatasetAnnotation & { dataset_id: string }> {
  const { data } = await client.put(`/annotations/${datasetId}`, ann)
  return data
}

export async function clearAnnotations(datasetId: string): Promise<void> {
  await client.delete(`/annotations/${datasetId}`)
}

// ── Schema validation ─────────────────────────────────────────────────────────

export interface TypeMismatch { feature: string; expected_type: string; actual_dtype: string }
export interface SchemaValidationResult {
  model_id: string
  target_col: string
  task_type: string
  n_rows: number
  n_expected_features: number
  schema_ok: boolean
  missing_cols: string[]
  extra_cols: string[]
  type_mismatches: TypeMismatch[]
  drift: {
    drifted_features: Array<{ feature: string; type: string; severity: string; psi?: number; mean_shift_std?: number; new_category_rate?: number }>
    n_drifted: number
    n_features_checked: number
    drift_rate: number
    overall_severity: 'none' | 'medium' | 'high'
  } | null
  lineage: {
    lineage_ok: boolean
    col_hash_match?: boolean
    columns_added?: string[]
    columns_removed?: string[]
    distribution_shifted?: string[]
    training_n_rows?: number
  } | null
}

export async function validateModelSchema(modelId: string, file: File): Promise<SchemaValidationResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await client.post<SchemaValidationResult>(`/models/${modelId}/validate-schema`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ── Observability ─────────────────────────────────────────────────────────────

export interface LatencyPhaseStats { avg_ms: number; p50_ms: number; p95_ms: number }
export interface LatencyStatsResponse { n_turns: number; phases: Record<string, LatencyPhaseStats> }
export async function getLatencyStats(): Promise<LatencyStatsResponse> {
  const { data } = await client.get<LatencyStatsResponse>('/health/latency')
  return data
}

export interface ScoringModelLatency { n: number; avg_ms: number; p50_ms: number; p95_ms: number }
export interface ScoringLatencyResponse { n_models: number; by_model: Record<string, ScoringModelLatency> }
export async function getScoringLatency(): Promise<ScoringLatencyResponse> {
  const { data } = await client.get<ScoringLatencyResponse>('/health/scoring-latency')
  return data
}

export interface PlannerFallbackResponse { total_fallbacks: number; by_reason: Record<string, number> }
export async function getPlannerFallbackRate(): Promise<PlannerFallbackResponse> {
  const { data } = await client.get<PlannerFallbackResponse>('/health/planner/fallback-rate')
  return data
}

export interface CorpusIndexStats { total_chunks: number; unique_sources: number; sources: string[] }
export async function getCorpusIndexStats(): Promise<CorpusIndexStats> {
  const { data } = await client.get<CorpusIndexStats>('/corpus/index-stats')
  return data
}
