import React, { useState, useEffect } from 'react'
import { LayoutDashboard } from 'lucide-react'
import type { ChatResponse, ToolResult } from '../types/api'
import DataTable from './DataTable'
import ChartView from './ChartView'

const ML_TOOL_NAMES = new Set([
  'train_supervised_model',
  'explain_model',
  'evaluate_ml_predictions',
  'score_with_model',
])

interface ResultsProps {
  response: ChatResponse | null
  conversationId: string | null
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
  const calibrated = trainResult.calibrated as boolean | undefined
  const optimalThreshold = trainResult.optimal_threshold as number | null | undefined
  const cv = trainResult.cv as { folds: number; scoring: string; mean: number; std: number } | undefined
  const modelComparison = trainResult.model_comparison as {
    previous_model_type: string
    metric: string
    previous: number
    current: number
    delta: number
    improved: boolean
  } | null | undefined
  const bestParams = trainResult.best_params as Record<string, unknown> | null | undefined
  const lagFeatureCols = trainResult.lag_feature_cols as string[] | undefined
  const interactionFeaturesAdded = trainResult.interaction_features_added as boolean | undefined
  const onnxExported = trainResult.onnx_exported as boolean | undefined
  const conformalHalfwidth = trainResult.conformal_halfwidth as number | null | undefined
  const piCoverage = trainResult.prediction_interval_coverage as number | null | undefined
  const baselineComparison = trainResult.baseline_comparison as {
    baselines: Record<string, Record<string, number>>
    primary_metric: string
    best_baseline_metric: number | null
    model_metric: number | null
    beats_baseline: boolean | null
    delta: number | null
  } | null | undefined
  const leakageWarnings = trainResult.leakage_warnings as Array<{
    feature: string
    risk: 'high' | 'medium'
    correlation: number | null
    reason: string
  }> | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Model Training</h3>

      {trainResult.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(trainResult.engineering_readout)}
        </div>
      )}

      {modelComparison != null && (
        <div className={`rounded-xl px-3.5 py-2.5 text-xs mb-3 border ${
          modelComparison.improved
            ? 'bg-green-50 border-green-200 text-green-800'
            : 'bg-rose-50 border-rose-200 text-rose-800'
        }`}>
          vs previous {modelComparison.previous_model_type}:{' '}
          {modelComparison.metric.toUpperCase()}{' '}
          {modelComparison.previous.toFixed(4)} → {modelComparison.current.toFixed(4)}{' '}
          <span className="font-semibold">
            ({modelComparison.improved ? '↑' : '↓'}{Math.abs(modelComparison.delta).toFixed(4)})
          </span>
        </div>
      )}

      {trainResult.model_id != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 mb-3">
          <div className="flex items-center justify-between mb-0.5">
            <p className="text-indigo-600 text-xs font-medium">Model ID</p>
            <div className="flex items-center gap-1.5">
              {interactionFeaturesAdded && (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-cyan-100 text-cyan-700"
                  title="Degree-2 pairwise interaction features were added">
                  ↔ interactions
                </span>
              )}
              {lagFeatureCols && lagFeatureCols.length > 0 && (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-teal-100 text-teal-700"
                  title={`Lag/rolling features: ${lagFeatureCols.join(', ')}`}>
                  ⏱ lag features
                </span>
              )}
              {onnxExported && (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-green-100 text-green-700"
                  title="ONNX artifact available — download from the model registry">
                  ONNX
                </span>
              )}
              {calibrated && (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-violet-100 text-violet-700">
                  Platt calibrated
                </span>
              )}
            </div>
          </div>
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
        {optimalThreshold != null && optimalThreshold !== 0.5 && (
          <MetricCard label="Decision threshold" value={`${optimalThreshold} (F1-opt)`} />
        )}
        {conformalHalfwidth != null && (
          <MetricCard
            label={`PI ±width (${piCoverage != null ? `${(piCoverage * 100).toFixed(0)}%` : '90%'})`}
            value={conformalHalfwidth.toFixed(4)}
          />
        )}
      </div>

      {cv && (
        <p className="text-slate-500 text-xs mb-3">
          CV {cv.folds}-fold {cv.scoring.replace('neg_', '')}:{' '}
          <span className="font-semibold text-slate-700">{Math.abs(cv.mean).toFixed(4)}</span>
          {' '}± {cv.std.toFixed(4)}
        </p>
      )}

      {/* Baseline comparison */}
      {baselineComparison != null && baselineComparison.best_baseline_metric != null && (
        <div className={`rounded-xl px-3.5 py-2.5 text-xs mb-3 border ${
          baselineComparison.beats_baseline === false
            ? 'bg-rose-50 border-rose-200 text-rose-800'
            : baselineComparison.beats_baseline === true
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
            : 'bg-slate-50 border-slate-200 text-slate-600'
        }`}>
          <div className="flex items-center justify-between mb-1">
            <span className="font-semibold">
              {baselineComparison.beats_baseline === false
                ? '⚠ Barely beats naive baseline'
                : baselineComparison.beats_baseline === true
                ? '✓ Beats naive baseline'
                : 'vs. naive baseline'}
            </span>
            {baselineComparison.delta != null && (
              <span className="font-mono font-semibold">
                {baselineComparison.delta > 0 ? '+' : ''}{baselineComparison.delta.toFixed(4)}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-3">
            {Object.entries(baselineComparison.baselines).map(([strategy, metrics]) => {
              const val = metrics[baselineComparison.primary_metric]
              return (
                <span key={strategy} className="opacity-80">
                  {strategy}: <span className="font-mono">{val?.toFixed(4) ?? '—'}</span>
                </span>
              )
            })}
            <span>
              model: <span className="font-mono font-semibold">{baselineComparison.model_metric?.toFixed(4) ?? '—'}</span>
            </span>
          </div>
        </div>
      )}

      {/* Leakage warnings */}
      {leakageWarnings && leakageWarnings.length > 0 && (
        <div className="bg-orange-50 border border-orange-300 rounded-xl px-3.5 py-2.5 text-xs text-orange-900 mb-3">
          <p className="font-semibold mb-1">Possible data leakage detected</p>
          <ul className="space-y-0.5">
            {leakageWarnings.map((w, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <span className={w.risk === 'high' ? 'text-red-600 font-bold' : 'text-orange-600 font-semibold'}>
                  {w.risk === 'high' ? '● high' : '◉ medium'}
                </span>
                <span>
                  <span className="font-mono">{w.feature}</span>
                  {w.correlation != null && <span className="ml-1 opacity-70">(r={w.correlation.toFixed(3)})</span>}
                  {w.reason === 'name_similarity' && <span className="ml-1 opacity-70">(name overlap)</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(trainResult.imbalance_ratio as number | undefined) != null &&
        Number(trainResult.imbalance_ratio) > 5 && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-3.5 py-2.5 text-xs text-amber-800 mb-3">
            Class imbalance detected ({Number(trainResult.imbalance_ratio).toFixed(1)}×) — balanced weights applied.
          </div>
        )}

      {(trainResult.preprocessing_notes as string[] | undefined)?.map((note, i) => (
        <div key={i} className="bg-blue-50 border border-blue-200 rounded-xl px-3.5 py-2.5 text-xs text-blue-800 mb-3">
          {note}
        </div>
      ))}

      {bestParams != null && Object.keys(bestParams).length > 0 && (
        <details className="mb-3 group">
          <summary className="text-slate-500 text-xs font-medium cursor-pointer select-none list-none flex items-center gap-1">
            <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
            HPO best params
          </summary>
          <div className="bg-white rounded-xl border border-slate-200 px-3 py-2 mt-1.5 divide-y divide-slate-50">
            {Object.entries(bestParams).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs py-1 first:pt-0 last:pb-0">
                <span className="text-slate-500 font-mono">{k}</span>
                <span className="text-slate-800 font-mono font-medium">{String(v)}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {lagFeatureCols && lagFeatureCols.length > 0 && (
        <details className="mb-3 group">
          <summary className="text-slate-500 text-xs font-medium cursor-pointer select-none list-none flex items-center gap-1">
            <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
            Auto-engineered lag columns ({lagFeatureCols.length})
          </summary>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {lagFeatureCols.map((col) => (
              <span
                key={col}
                className="font-mono text-xs bg-teal-50 border border-teal-200 text-teal-800 rounded px-1.5 py-0.5"
              >
                {col}
              </span>
            ))}
          </div>
        </details>
      )}

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

  const drift = scoreResult.drift as {
    drifted_features: Array<{
      feature: string
      type: string
      severity: string
      mean_shift_std?: number
      std_ratio?: number
      new_category_rate?: number
      missing_rate_delta?: number
    }>
    n_drifted: number
    n_features_checked: number
    drift_rate: number
    overall_severity: 'none' | 'medium' | 'high'
  } | null | undefined

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
        {(scoreResult.conformal_halfwidth as number | null | undefined) != null && (
          <MetricCard
            label="PI ±width (90%)"
            value={Number(scoreResult.conformal_halfwidth).toFixed(4)}
          />
        )}
      </div>

      {(scoreResult.conformal_halfwidth as number | null | undefined) != null && (
        <p className="text-slate-500 text-xs mb-3">
          Columns <span className="font-mono text-slate-400">prediction_lower_90</span> and{' '}
          <span className="font-mono text-slate-400">prediction_upper_90</span> added to scored output.
        </p>
      )}

      {drift && drift.overall_severity !== 'none' && (
        <div className={`rounded-xl px-3.5 py-2.5 text-xs mb-3 border ${
          drift.overall_severity === 'high'
            ? 'bg-rose-50 border-rose-200 text-rose-800'
            : 'bg-amber-50 border-amber-200 text-amber-800'
        }`}>
          <p className="font-semibold mb-1">
            Feature drift detected — {drift.n_drifted}/{drift.n_features_checked} features shifted
          </p>
          <ul className="space-y-0.5">
            {drift.drifted_features.slice(0, 8).map((f, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <span className={`font-semibold flex-shrink-0 ${f.severity === 'high' ? 'text-red-600' : 'text-amber-600'}`}>
                  {f.severity === 'high' ? '● high' : '◉ med'}
                </span>
                <span>
                  <span className="font-mono">{f.feature}</span>
                  {f.type === 'numeric' && f.mean_shift_std != null && (
                    <span className="ml-1 opacity-70">mean shift {f.mean_shift_std.toFixed(1)}σ</span>
                  )}
                  {f.type === 'categorical' && f.new_category_rate != null && (
                    <span className="ml-1 opacity-70">{(f.new_category_rate * 100).toFixed(0)}% unseen categories</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
          {drift.drifted_features.length > 8 && (
            <p className="mt-1 opacity-60">…and {drift.drifted_features.length - 8} more</p>
          )}
        </div>
      )}
    </div>
  )
}

const Results = React.memo(function Results({ response, conversationId }: ResultsProps) {
  const [mlResults, setMlResults] = useState<ToolResult[]>([])

  // Clear dashboard when the user starts a new conversation
  useEffect(() => {
    setMlResults([])
  }, [conversationId])

  // Merge incoming ML tool results into the persistent dashboard,
  // replacing the previous result for each tool name so each card
  // shows the latest run while surviving follow-up queries.
  useEffect(() => {
    if (!response) return
    const incoming = response.tool_results.filter(r => ML_TOOL_NAMES.has(r.name) && r.ok)
    if (incoming.length === 0) return
    setMlResults(prev => {
      const map = new Map(prev.map(r => [r.name, r]))
      for (const r of incoming) map.set(r.name, r)
      return Array.from(map.values())
    })
  }, [response])

  const hasMlDashboard = mlResults.length > 0
  const hasTables = (response?.tables.length ?? 0) > 0
  const hasCharts = (response?.charts.length ?? 0) > 0
  const hasLatestData = hasTables || hasCharts
  const isEmpty = !hasMlDashboard && !hasLatestData

  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-slate-50">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200 bg-white">
        <LayoutDashboard size={16} className="text-slate-500" />
        <h2 className="text-slate-800 font-semibold text-sm">Results</h2>
      </div>

      <div className="flex-1 overflow-y-auto thin-scroll px-4 py-4 space-y-4">
        {isEmpty ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-400 text-sm text-center">
              Upload a dataset and run a query to see results.
            </p>
          </div>
        ) : (
          <>
            {/* ML dashboard — persists across turns until new conversation */}
            {hasMlDashboard && (
              <>
                <MLEvalSummary results={mlResults} />
                <MLTrainSummary results={mlResults} />
                <MLExplainSummary results={mlResults} />
                <MLScoreSummary results={mlResults} />
              </>
            )}

            {/* Divider between persistent ML cards and latest query data */}
            {hasMlDashboard && hasLatestData && (
              <div className="flex items-center gap-2 py-1">
                <div className="flex-1 h-px bg-slate-200" />
                <span className="text-slate-400 text-xs font-medium">Latest query</span>
                <div className="flex-1 h-px bg-slate-200" />
              </div>
            )}

            {/* Tables and charts — always the most recent response */}
            {response?.tables.map((table, i) => (
              <DataTable
                key={`${table.title}-${i}`}
                title={table.title}
                columns={table.columns}
                data={table.data}
              />
            ))}
            {response?.charts.map((chart, i) => (
              <ChartView key={`${chart.title}-${i}`} chart={chart} />
            ))}
          </>
        )}
      </div>
    </div>
  )
})

export default Results
