import React from 'react'
import { LayoutDashboard } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatResponse, ToolResult } from '../types/api'
import DataTable from './DataTable'
import ChartView from './ChartView'

interface ResultsProps {
  response: ChatResponse | null
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-4 py-3">
      <p className="text-slate-500 text-xs font-medium mb-1">{label}</p>
      <p className="text-slate-900 text-xl font-semibold">{value}</p>
    </div>
  )
}

interface FeatureImportanceRow {
  feature: string
  // Training importance path (pipeline built-in)
  importance?: number
  importance_mean?: number
  importance_std?: number
  // SHAP / permutation path (explain_model)
  shap_mean_abs?: number
}

function FeatureImportanceTable({
  rows,
  title = 'Feature importance',
  method,
}: {
  rows: FeatureImportanceRow[] | undefined
  title?: string
  method?: string
}) {
  if (!rows || rows.length === 0) return null

  const maxVal = Math.max(
    ...rows.map((r) => r.shap_mean_abs ?? r.importance ?? r.importance_mean ?? 0)
  )

  const methodBadge: Record<string, { label: string; cls: string }> = {
    shap_tree:    { label: 'SHAP tree',    cls: 'bg-violet-100 text-violet-700' },
    shap_linear:  { label: 'SHAP linear',  cls: 'bg-violet-100 text-violet-700' },
    permutation:  { label: 'permutation',  cls: 'bg-slate-100 text-slate-600'   },
  }
  const badge = method ? methodBadge[method] : undefined

  return (
    <div className="mb-3">
      <div className="flex items-center gap-2 mb-1.5">
        <p className="text-slate-600 text-xs font-semibold uppercase tracking-wide">{title}</p>
        {badge && (
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge.cls}`}>
            {badge.label}
          </span>
        )}
      </div>
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left px-3 py-2 text-slate-500 font-medium w-1/3">Feature</th>
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Importance</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium w-20">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const val = row.shap_mean_abs ?? row.importance ?? row.importance_mean ?? 0
              const pct = maxVal > 0 ? (val / maxVal) * 100 : 0
              return (
                <tr key={i} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                  <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[140px]">
                    {row.feature}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-100 rounded-full h-1.5 min-w-[60px]">
                        <div
                          className="bg-indigo-500 h-1.5 rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right text-slate-600 font-mono tabular-nums">
                    {val.toFixed(4)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MLExplainSummary({ results }: { results: ToolResult[] }) {
  const explainResult = results.find(
    (r) => r.name === 'explain_model' && r.ok
  )?.result as Record<string, unknown> | undefined

  if (!explainResult || 'error' in explainResult) return null

  const method = explainResult.method as string | undefined
  const rows = explainResult.feature_importances as FeatureImportanceRow[] | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Feature Importance</h3>
      {explainResult.engineering_readout != null && (
        <div className="bg-violet-50 border border-violet-200 rounded-xl px-3.5 py-2.5 text-sm text-violet-800 mb-3">
          {String(explainResult.engineering_readout)}
        </div>
      )}
      <FeatureImportanceTable rows={rows} title="Feature scores" method={method} />
    </div>
  )
}

function MLEvalSummary({ results }: { results: ToolResult[] }) {
  const evalResult = results.find(
    (r) => r.name === 'evaluate_ml_predictions' && r.ok
  )?.result as Record<string, unknown> | undefined

  if (!evalResult) return null

  const evaluation = evalResult.evaluation as Record<string, unknown> | undefined
  const scoreSummary = evaluation?.score_summary as Record<string, unknown> | undefined
  const confBands = evaluation?.confidence_bands as Record<string, unknown> | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Model Evaluation</h3>
      {evalResult.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(evalResult.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows scored" value={(evalResult.n_rows_scored as number | undefined ?? 0).toLocaleString()} />
        <MetricCard
          label="Mean score"
          value={scoreSummary?.mean != null ? Number(scoreSummary.mean).toFixed(4) : 'N/A'}
        />
        <MetricCard
          label="P95 score"
          value={scoreSummary?.p95 != null ? Number(scoreSummary.p95).toFixed(4) : 'N/A'}
        />
        <MetricCard
          label="High confidence"
          value={confBands?.['high_confidence_0_80_plus'] != null
            ? Number(confBands['high_confidence_0_80_plus']).toLocaleString()
            : 'N/A'}
        />
      </div>
    </div>
  )
}

function MLTrainSummary({ results }: { results: ToolResult[] }) {
  const trainResult = results.find(
    (r) => r.name === 'train_supervised_model' && r.ok
  )?.result as Record<string, unknown> | undefined

  if (!trainResult || 'error' in trainResult) return null

  const evaluation = trainResult.evaluation as Record<string, unknown> | undefined
  const taskType = trainResult.task_type as string | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Model Training</h3>
      {trainResult.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(trainResult.engineering_readout)}
        </div>
      )}
      {trainResult.model_id != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 mb-3">
          <p className="text-indigo-600 text-xs font-medium mb-0.5">Model ID</p>
          <p className="text-indigo-900 text-xs font-mono break-all">{String(trainResult.model_id)}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Task" value={taskType ?? 'N/A'} />
        <MetricCard
          label={taskType === 'classification' ? 'Accuracy' : 'WMAPE'}
          value={
            taskType === 'classification'
              ? evaluation?.accuracy != null ? Number(evaluation.accuracy).toFixed(4) : 'N/A'
              : evaluation?.wmape != null ? Number(evaluation.wmape).toFixed(4) : 'N/A'
          }
        />
        <MetricCard label="Train rows" value={(trainResult.n_rows_train as number | undefined ?? 0).toLocaleString()} />
        <MetricCard label="Test rows" value={(trainResult.n_rows_test as number | undefined ?? 0).toLocaleString()} />
      </div>
      {(trainResult.imbalance_ratio as number | undefined) != null &&
        Number(trainResult.imbalance_ratio) > 5 && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-3.5 py-2.5 text-xs text-amber-800 mb-3">
            Class imbalance detected ({Number(trainResult.imbalance_ratio).toFixed(1)}×) — class_weight=balanced applied.
          </div>
        )}
      {(trainResult.preprocessing_notes as string[] | undefined)?.map((note, i) => (
        <div key={i} className="bg-blue-50 border border-blue-200 rounded-xl px-3.5 py-2.5 text-xs text-blue-800 mb-3">
          {note}
        </div>
      ))}
      <FeatureImportanceTable
        rows={trainResult.feature_importance as FeatureImportanceRow[] | undefined}
        title="Top features (training)"
      />
    </div>
  )
}

function MLScoreSummary({ results }: { results: ToolResult[] }) {
  const scoreResult = results.find(
    (r) => r.name === 'score_with_model' && r.ok
  )?.result as Record<string, unknown> | undefined

  if (!scoreResult || 'error' in scoreResult) return null

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Scoring</h3>
      {scoreResult.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(scoreResult.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows scored" value={(scoreResult.n_rows_scored as number | undefined ?? 0).toLocaleString()} />
        <MetricCard label="Task" value={(scoreResult.task_type as string | undefined) ?? 'N/A'} />
      </div>
    </div>
  )
}

const Results = React.memo(function Results({ response }: ResultsProps) {
  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-slate-50">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200 bg-white">
        <LayoutDashboard size={16} className="text-slate-500" />
        <h2 className="text-slate-800 font-semibold text-sm">Results</h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto thin-scroll px-4 py-4 space-y-4">
        {!response ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-400 text-sm text-center">
              Upload a dataset and run a query to see results.
            </p>
          </div>
        ) : (
          <>
            {/* Assistant narrative — always shown first */}
            {response.message && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-slate-500 text-xs font-medium uppercase tracking-wide">Analysis</p>
                  {response.llm_enabled && (
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        response.synthesis_source === 'llm'
                          ? 'bg-indigo-100 text-indigo-700'
                          : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {response.synthesis_source === 'llm' ? 'LLM synthesis' : 'Rule-based'}
                    </span>
                  )}
                </div>
                <div className="text-slate-800 text-sm leading-relaxed prose-sm">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                      ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-2">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-2">{children}</ol>,
                      li: ({ children }) => <li>{children}</li>,
                      strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
                      h2: ({ children }) => <h2 className="font-semibold text-sm mt-3 mb-1">{children}</h2>,
                      h3: ({ children }) => <h3 className="font-medium text-sm mt-2 mb-1">{children}</h3>,
                      code: ({ children }) => (
                        <code className="bg-slate-100 text-indigo-700 px-1 py-0.5 rounded text-xs font-mono">
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {response.message}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {/* ML summaries */}
            {response.tool_results.length > 0 && (
              <>
                <MLEvalSummary results={response.tool_results} />
                <MLTrainSummary results={response.tool_results} />
                <MLExplainSummary results={response.tool_results} />
                <MLScoreSummary results={response.tool_results} />
              </>
            )}

            {/* Tables */}
            {response.tables.map((table, i) => (
              <DataTable
                key={`${table.title}-${i}`}
                title={table.title}
                columns={table.columns}
                data={table.data}
              />
            ))}

            {/* Charts */}
            {response.charts.map((chart, i) => (
              <ChartView key={`${chart.title}-${i}`} chart={chart} />
            ))}
          </>
        )}
      </div>
    </div>
  )
})

export default Results
