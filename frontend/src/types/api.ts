export interface UploadResponse {
  dataset_id: string
  filename: string
  n_rows: number
  n_cols: number
  notes: string[]
}

export interface Dataset {
  dataset_id: string
  filename: string
  n_rows: number
  n_cols: number
}

export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
}

export interface ToolResult {
  name: string
  ok: boolean
  result?: unknown
  error?: string
}

export interface TableSpec {
  title: string
  columns: string[]
  data: Record<string, unknown>[]
}

export interface ChartSpec {
  type: 'bar' | 'line' | 'scatter' | 'histogram'
  title: string
  x: string
  y?: string
  y_series?: string[]
  data: Record<string, unknown>[]
  correlation?: number
  x_label?: string
  column?: string
}

export interface Citation {
  source_id: string
  score: number
  text: string
}

export interface ChatResponse {
  dataset_id: string | null
  conversation_id: string
  message: string
  tool_calls: ToolCall[]
  tool_results: ToolResult[]
  tables: TableSpec[]
  charts: ChartSpec[]
  citations: Citation[]
  llm_enabled: boolean
  planning_source: 'llm' | 'rules'
  synthesis_source: 'llm' | 'rules'
  llm_error?: string
  llm_notes: string[]
  groundedness_score?: number
  groundedness_criteria: Record<string, number>
  groundedness_issues: string[]
  judge_status: 'judged' | 'not_sampled' | 'rule_based' | 'llm_disabled' | 'failed'
}

export interface ConversationTurn {
  role: 'user' | 'assistant'
  content: string
  dataset_id?: string
  tool_calls: Record<string, unknown>[]
  tool_results: ToolResult[]
  timestamp: number
  tables: TableSpec[]
  charts: ChartSpec[]
  groundedness_score?: number
  groundedness_criteria: Record<string, number>
  groundedness_issues: string[]
  judge_status: 'judged' | 'not_sampled' | 'rule_based' | 'llm_disabled' | 'failed'
  planning_source: string
  synthesis_source: string
  citations: Citation[]
}

export interface SSEPlanEvent {
  type: 'plan'
  tool_calls: ToolCall[]
  conversation_id: string
}

export interface SSEToolResultEvent {
  type: 'tool_result'
  name: string
  ok: boolean
  error?: string
}

export interface SSEErrorEvent {
  type: 'error'
  detail: string
}

export interface SSEDoneEvent {
  type: 'done'
  response: ChatResponse
}

export interface SSEThinkingEvent {
  type: 'thinking'
}

export interface SSESynthesizingEvent {
  type: 'synthesizing'
}

export type SSEEvent =
  | SSEThinkingEvent
  | SSESynthesizingEvent
  | SSEPlanEvent
  | SSEToolResultEvent
  | SSEErrorEvent
  | SSEDoneEvent

export interface LLMOperationStats {
  count: number
  errors: number
  avg_latency_ms: number
}

export interface LLMStats {
  window_size: number
  error_count: number
  error_rate: number
  avg_latency_ms: number
  total_tokens_sampled: number
  by_operation: Record<string, LLMOperationStats>
}

export interface RagEvalQueryResult {
  query: string
  expected_source: string
  hit: boolean
  rank: number | null
  top_sources: string[]
  score: number | null
}

export interface RagEvalResponse {
  available: boolean
  n_queries: number
  recall_at_1?: number
  recall_at_3?: number
  recall_at_5?: number
  mrr?: number
  generated_at?: number
  queries: RagEvalQueryResult[]
}

export interface Model {
  model_id: string
  model_type: string
  task_type: string
  target_col: string
  feature_cols: string[]
  log_transform_target: boolean
  lag_config?: Record<string, unknown> | null
  onnx_path?: string | null
  created_at: string
}

export interface ExperimentMetrics {
  accuracy?: number
  wmape?: number
  r2?: number
  cv_mean?: number | null
  cv_std?: number | null
  optimal_threshold?: number | null
  calibrated?: boolean
  [key: string]: unknown
}

export interface ExperimentParams {
  model_type?: string
  feature_cols?: string[]
  best_params?: Record<string, unknown> | null
  add_interactions?: boolean
  lag_config?: Record<string, unknown> | null
  [key: string]: unknown
}

export interface Experiment {
  run_id: string
  model_id: string
  dataset_id: string | null
  target_col: string
  task_type: string
  model_type: string
  params: ExperimentParams
  metrics: ExperimentMetrics
  preprocessing: Record<string, unknown>
  comparison: Record<string, unknown> | null
  created_at: string
}

export interface ToolProgress {
  name: string
  status: 'pending' | 'ok' | 'error'
  error?: string
}

export interface TrainingJob {
  job_id: string
  status: 'running' | 'done' | 'error'
  created_at: string
  completed_at?: string | null
  result?: Record<string, unknown> | null
  error?: string | null
}

export interface LineageReport {
  lineage_ok: boolean
  col_hash_match: boolean
  columns_added: string[]
  columns_removed: string[]
  distribution_shifted: string[]
  training_n_rows: number | null
}

export interface PredictionSetInfo {
  coverage_target: number
  threshold: number
  avg_set_size: number
  n_singleton: number
}

export interface JudgeStats {
  response_count: number
  eligible_count: number
  attempted_count: number
  sampled_count: number
  skipped_count: number
  skipped_sample_rate_count: number
  skipped_rule_based_count: number
  skipped_llm_disabled_count: number
  error_count: number
  avg_groundedness_score: number
  low_score_rate: number
  flagged_rate: number
  last_error?: string | null
}

export interface JudgeHistoryEntry {
  score: number
  issue_count: number
  synthesis_source: string
  timestamp: number
}

export interface JudgeHistoryResponse {
  entries: JudgeHistoryEntry[]
  total: number
}
