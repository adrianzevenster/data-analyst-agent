import axios from 'axios'
import type {
  Dataset,
  ConversationTurn,
  Experiment,
  Model,
  LLMStats,
  RagEvalResponse,
  TrainingJob,
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
