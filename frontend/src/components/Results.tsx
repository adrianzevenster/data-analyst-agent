import React, { useState, useEffect } from 'react'
import { LayoutDashboard } from 'lucide-react'
import type { ChatResponse, ChartSpec, Experiment, LineageReport, PredictionSetInfo, ToolResult } from '../types/api'
import { getExperiments } from '../lib/api'
import DataTable from './DataTable'
import ChartView from './ChartView'

const ML_TOOL_NAMES = new Set([
  'train_supervised_model',
  'explain_model',
  'shap_explain_prediction',
  'evaluate_ml_predictions',
  'evaluate_trained_model',
  'score_with_model',
  'forecast_with_model',
  'compute_pdp',
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

function ClassificationReportTable({
  report,
}: {
  report: Record<string, Record<string, number>> | undefined
}) {
  if (!report) return null
  const classRows = Object.entries(report).filter(
    ([k]) => !['accuracy', 'macro avg', 'weighted avg'].includes(k)
  )
  if (classRows.length === 0) return null
  return (
    <div className="mb-3">
      <p className="text-slate-600 text-xs font-semibold uppercase tracking-wide mb-1.5">Per-class metrics</p>
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Class</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">Precision</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">Recall</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">F1</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">Support</th>
            </tr>
          </thead>
          <tbody>
            {classRows.map(([cls, m]) => (
              <tr key={cls} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                <td className="px-3 py-1.5 font-mono text-slate-800">{cls}</td>
                <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-700">
                  {m.precision != null ? m.precision.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-700">
                  {m.recall != null ? m.recall.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-right font-mono tabular-nums font-semibold text-indigo-700">
                  {m['f1-score'] != null ? m['f1-score'].toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-right text-slate-400">
                  {m.support != null ? Number(m.support).toLocaleString() : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const maxCount = Math.max(...matrix.flat(), 1)
  return (
    <div className="mb-3">
      <p className="text-slate-600 text-xs font-semibold uppercase tracking-wide mb-1.5">Confusion matrix</p>
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-3 overflow-x-auto">
        <table className="text-xs border-separate border-spacing-1">
          <thead>
            <tr>
              <th className="text-slate-400 font-normal pr-2 pb-1 text-right whitespace-nowrap">actual ↓ / pred →</th>
              {labels.map((l) => (
                <th key={l} className="text-slate-600 font-semibold px-3 pb-1 text-center font-mono min-w-[52px]">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="text-slate-600 font-semibold pr-2 text-right font-mono">{labels[ri]}</td>
                {row.map((count, ci) => {
                  const isDiag = ri === ci
                  const intensity = count / maxCount
                  return (
                    <td
                      key={ci}
                      className="text-center rounded-md px-3 py-2 font-mono font-semibold"
                      style={{
                        backgroundColor: isDiag
                          ? `rgba(16,185,129,${Math.max(0.08, intensity * 0.7)})`
                          : count > 0 ? `rgba(239,68,68,${Math.max(0.06, intensity * 0.55)})` : 'transparent',
                        color: isDiag
                          ? intensity > 0.5 ? '#064e3b' : '#065f46'
                          : count > 0 ? (intensity > 0.5 ? '#7f1d1d' : '#991b1b') : '#94a3b8',
                      }}
                    >
                      {count.toLocaleString()}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MLEvalCharts({ evaluation }: { evaluation: Record<string, unknown> | undefined }) {
  if (!evaluation) return null

  const rocCurve = evaluation.roc_curve as ChartSpec | undefined
  const prCurve = evaluation.pr_curve as ChartSpec | undefined
  const calibCurve = evaluation.calibration_curve as ChartSpec | undefined
  const actualVsPred = evaluation.actual_vs_predicted as ChartSpec | undefined
  const residualsHist = evaluation.residuals_hist as ChartSpec | undefined
  const confMatrix = evaluation.confusion_matrix as { labels: unknown[]; matrix: number[][] } | undefined
  const classReport = evaluation.classification_report as Record<string, Record<string, number>> | undefined

  const hasContent = rocCurve || prCurve || calibCurve || actualVsPred || residualsHist || confMatrix || classReport
  if (!hasContent) return null

  return (
    <div className="space-y-3 mb-3">
      <ClassificationReportTable report={classReport} />
      {confMatrix?.labels && confMatrix?.matrix && (
        <ConfusionMatrix labels={confMatrix.labels.map(String)} matrix={confMatrix.matrix} />
      )}
      {(rocCurve || prCurve) && (
        <div className={`grid gap-3 ${rocCurve && prCurve ? 'grid-cols-2' : 'grid-cols-1'}`}>
          {rocCurve && <ChartView chart={rocCurve} />}
          {prCurve && <ChartView chart={prCurve} />}
        </div>
      )}
      {calibCurve && <ChartView chart={calibCurve} />}
      {actualVsPred && <ChartView chart={actualVsPred} />}
      {residualsHist && <ChartView chart={residualsHist} />}
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

function MLStoredEvalSummary({ results }: { results: ToolResult[] }) {
  const evalResult = results.find(
    (r) => r.name === 'evaluate_trained_model' && r.ok
  )?.result as Record<string, unknown> | undefined

  if (!evalResult || 'error' in evalResult) return null

  const evaluation = evalResult.evaluation as Record<string, unknown> | undefined
  const taskType = evalResult.task_type as string | undefined
  const modelType = evalResult.model_type as string | undefined
  const targetCol = evalResult.target_col as string | undefined
  const optimalThreshold = evalResult.optimal_threshold as number | null | undefined
  const conformalHalfwidth = evalResult.conformal_halfwidth as number | null | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Trained Model Evaluation</h3>
      {evalResult.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(evalResult.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Task" value={taskType ?? 'N/A'} />
        <MetricCard label="Model" value={modelType ?? 'N/A'} />
        <MetricCard label="Target" value={targetCol ?? 'N/A'} />
        <MetricCard
          label={taskType === 'classification' ? 'Accuracy' : 'WMAPE'}
          value={
            taskType === 'classification'
              ? evaluation?.accuracy != null ? Number(evaluation.accuracy).toFixed(4) : 'N/A'
              : evaluation?.wmape != null ? Number(evaluation.wmape).toFixed(4) : 'N/A'
          }
        />
        {evaluation?.f1 != null && (
          <MetricCard label="F1" value={Number(evaluation.f1).toFixed(4)} />
        )}
        {evaluation?.roc_auc != null && (
          <MetricCard label="ROC AUC" value={Number(evaluation.roc_auc).toFixed(4)} />
        )}
        {evaluation?.r2 != null && (
          <MetricCard label="R2" value={Number(evaluation.r2).toFixed(4)} />
        )}
        {evaluation?.rmse != null && (
          <MetricCard label="RMSE" value={Number(evaluation.rmse).toFixed(4)} />
        )}
        {optimalThreshold != null && optimalThreshold !== 0.5 && (
          <MetricCard label="Decision threshold" value={`${optimalThreshold} (F1-opt)`} />
        )}
        {conformalHalfwidth != null && (
          <MetricCard label="PI ±width (90%)" value={conformalHalfwidth.toFixed(4)} />
        )}
      </div>
      <MLEvalCharts evaluation={evaluation} />
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

      <MLEvalCharts evaluation={evaluation} />
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

  const lineage = scoreResult.lineage as LineageReport | null | undefined
  const predSetInfo = scoreResult.prediction_set_info as PredictionSetInfo | null | undefined

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-slate-700 font-semibold text-sm">Scoring</h3>
        <button
          onClick={() => {
            const rows = scoreResult.scored_rows as Record<string, unknown>[] | undefined
            if (!rows || rows.length === 0) return
            const keys = Object.keys(rows[0])
            const header = keys.join(',')
            const body = rows.map((r) =>
              keys.map((k) => {
                const v = r[k]
                if (v == null) return ''
                const s = String(v)
                return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replace(/"/g, '""')}"` : s
              }).join(',')
            )
            const csv = [header, ...body].join('\n')
            const a = document.createElement('a')
            a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
            a.download = `predictions__${String(scoreResult.model_id ?? 'model').slice(0, 8)}.csv`
            a.click()
          }}
          className="text-xs text-indigo-600 hover:text-indigo-800 font-medium underline transition-colors"
        >
          ↓ Download CSV
        </button>
      </div>
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

      {/* Conformal prediction sets */}
      {predSetInfo && (
        <div className="bg-violet-50 border border-violet-200 rounded-xl px-3.5 py-2.5 text-xs text-violet-900 mb-3">
          <div className="flex items-center justify-between mb-0.5">
            <span className="font-semibold">Prediction sets (90% conformal coverage)</span>
            <span className="font-mono font-semibold">avg size {predSetInfo.avg_set_size.toFixed(1)}</span>
          </div>
          <p className="opacity-80">
            <span className="font-mono">prediction_set</span> column added —{' '}
            {predSetInfo.n_singleton.toLocaleString()} singleton predictions,{' '}
            threshold {predSetInfo.threshold.toFixed(3)}.
          </p>
        </div>
      )}

      {/* Data lineage */}
      {lineage && (
        <div className={`rounded-xl px-3.5 py-2.5 text-xs mb-3 border ${
          lineage.lineage_ok
            ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
            : 'bg-amber-50 border-amber-200 text-amber-900'
        }`}>
          <div className="flex items-center justify-between mb-0.5">
            <span className="font-semibold">
              {lineage.lineage_ok ? '✓ Data matches training' : '⚠ Data changed since training'}
            </span>
            {lineage.training_n_rows != null && (
              <span className="opacity-70">trained on {lineage.training_n_rows.toLocaleString()} rows</span>
            )}
          </div>
          {!lineage.lineage_ok && (
            <div className="space-y-0.5 mt-1">
              {lineage.columns_removed.length > 0 && (
                <p>Columns removed: <span className="font-mono">{lineage.columns_removed.join(', ')}</span></p>
              )}
              {lineage.columns_added.length > 0 && (
                <p>Columns added: <span className="font-mono">{lineage.columns_added.join(', ')}</span></p>
              )}
              {lineage.distribution_shifted.length > 0 && (
                <p>Distribution shifted (&gt;2σ): <span className="font-mono">{lineage.distribution_shifted.join(', ')}</span></p>
              )}
            </div>
          )}
        </div>
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

      {/* Retrain CTA — shown only when drift is high severity */}
      {drift && drift.overall_severity === 'high' && (
        <div className="bg-amber-50 border border-amber-300 rounded-xl px-3.5 py-2.5 text-xs text-amber-900 mb-3">
          <div className="flex items-center justify-between mb-0.5">
            <span className="font-semibold">↻ Retrain recommended</span>
            <button
              onClick={() => {
                const target = (scoreResult.target_col as string | undefined) ?? ''
                const prompt = `Train a new ${scoreResult.task_type ?? 'regression'} model to predict ${target}`
                navigator.clipboard.writeText(prompt)
              }}
              className="text-amber-700 hover:text-amber-900 underline transition-colors"
            >
              Copy prompt
            </button>
          </div>
          <p className="opacity-80">
            High drift on {drift.n_drifted} features — current data has shifted significantly from training distribution.
            Retrain on updated data to restore reliability.
          </p>
        </div>
      )}
    </div>
  )
}

function MLForecastSummary({ results }: { results: ToolResult[] }) {
  const r = results.find((t) => t.name === 'forecast_with_model' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null

  const rows = r.forecast_rows as Array<{ step: number; date: string; prediction: number; lower_90?: number; upper_90?: number }> | undefined
  const hasPi = r.has_prediction_intervals as boolean | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Forecast</h3>
      {r.engineering_readout != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 text-sm text-indigo-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Model" value={(r.model_type as string | undefined) ?? 'N/A'} />
        <MetricCard label="Target" value={(r.target_col as string | undefined) ?? 'N/A'} />
        <MetricCard label="Horizon steps" value={(r.horizon_steps as number | undefined ?? 0).toLocaleString()} />
        {hasPi && (
          <MetricCard label="PI ±width (90%)" value={Number(r.conformal_halfwidth ?? 0).toFixed(4)} />
        )}
      </div>
      {rows && rows.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Step</th>
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Date</th>
                <th className="text-right px-3 py-2 text-slate-500 font-medium">Prediction</th>
                {hasPi && <th className="text-right px-3 py-2 text-slate-500 font-medium">90% PI</th>}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 10).map((row) => (
                <tr key={row.step} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                  <td className="px-3 py-1.5 text-slate-500">{row.step}</td>
                  <td className="px-3 py-1.5 font-mono text-slate-700">{row.date}</td>
                  <td className="px-3 py-1.5 text-right font-mono font-semibold text-indigo-700">
                    {row.prediction.toFixed(2)}
                  </td>
                  {hasPi && row.lower_90 != null && (
                    <td className="px-3 py-1.5 text-right text-slate-500 font-mono text-xs">
                      [{row.lower_90.toFixed(2)}, {row.upper_90?.toFixed(2)}]
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 10 && (
            <p className="px-3 py-1.5 text-slate-400 text-xs">…and {rows.length - 10} more steps</p>
          )}
        </div>
      )}
    </div>
  )
}

function MLExplainPredictionSummary({ results }: { results: ToolResult[] }) {
  const r = results.find((t) => t.name === 'shap_explain_prediction' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null

  const contribs = r.feature_contributions as Array<{ feature: string; shap_value: number }> | undefined
  if (!contribs || contribs.length === 0) return null

  const maxAbs = Math.max(...contribs.map((c) => Math.abs(c.shap_value)))
  const rowIdx = r.row_idx as number | undefined
  const pred = r.prediction as string | undefined
  const prob = r.prediction_probability as number | null | undefined
  const baseVal = r.shap_base_value as number | undefined
  const method = r.method as string | undefined

  const methodBadge: Record<string, { label: string; cls: string }> = {
    shap_tree:   { label: 'SHAP tree',   cls: 'bg-violet-100 text-violet-700' },
    shap_linear: { label: 'SHAP linear', cls: 'bg-violet-100 text-violet-700' },
  }
  const badge = method ? methodBadge[method] : undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Local Prediction Explanation</h3>
      {r.engineering_readout != null && (
        <div className="bg-violet-50 border border-violet-200 rounded-xl px-3.5 py-2.5 text-sm text-violet-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {rowIdx != null && <MetricCard label="Row" value={rowIdx} />}
        {pred != null && <MetricCard label="Prediction" value={pred} />}
        {prob != null && <MetricCard label="Probability" value={prob.toFixed(3)} />}
        {baseVal != null && <MetricCard label="Base value" value={baseVal.toFixed(4)} />}
      </div>
      <div className="flex items-center gap-2 mb-1.5">
        <p className="text-slate-600 text-xs font-semibold uppercase tracking-wide">Feature contributions</p>
        {badge && (
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge.cls}`}>{badge.label}</span>
        )}
      </div>
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left px-3 py-2 text-slate-500 font-medium w-1/3">Feature</th>
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Contribution</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium w-20">SHAP</th>
            </tr>
          </thead>
          <tbody>
            {contribs.map((c, i) => {
              const pct = maxAbs > 0 ? (Math.abs(c.shap_value) / maxAbs) * 100 : 0
              const positive = c.shap_value >= 0
              return (
                <tr key={i} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                  <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[140px]">{c.feature}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-1">
                      {/* Centred waterfall bar */}
                      <div className="flex-1 flex items-center min-w-[60px]">
                        {positive ? (
                          <>
                            <div className="w-1/2" />
                            <div className="bg-emerald-400 h-1.5 rounded-r" style={{ width: `${pct / 2}%` }} />
                          </>
                        ) : (
                          <>
                            <div className="flex-1 flex justify-end">
                              <div className="bg-rose-400 h-1.5 rounded-l" style={{ width: `${pct / 2}%` }} />
                            </div>
                            <div className="w-1/2" />
                          </>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${positive ? 'text-emerald-600' : 'text-rose-600'}`}>
                    {c.shap_value >= 0 ? '+' : ''}{c.shap_value.toFixed(4)}
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

function MLPDPSummary({ results }: { results: ToolResult[] }) {
  const r = results.find((t) => t.name === 'compute_pdp' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null

  const nPlotted = r.n_features_plotted as number | undefined
  const features = r.feature_cols as string[] | undefined
  const sampleSize = r.sample_size as number | undefined

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Partial Dependence</h3>
      {r.engineering_readout != null && (
        <div className="bg-violet-50 border border-violet-200 rounded-xl px-3.5 py-2.5 text-sm text-violet-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Features plotted" value={nPlotted ?? 0} />
        {sampleSize != null && <MetricCard label="Sample size" value={sampleSize.toLocaleString()} />}
      </div>
      {features && features.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {features.map((f) => (
            <span key={f} className="font-mono text-xs bg-violet-50 border border-violet-200 text-violet-800 rounded px-1.5 py-0.5">
              {f}
            </span>
          ))}
        </div>
      )}
      <p className="text-slate-400 text-xs mt-2">Charts appear in the Latest query section below.</p>
    </div>
  )
}

function MLComparisonPanel({
  results,
  datasetId,
}: {
  results: ToolResult[]
  datasetId: string | null
}) {
  const trainResult = results.find((r) => r.name === 'train_supervised_model' && r.ok)
    ?.result as Record<string, unknown> | undefined
  const targetCol = trainResult?.target_col as string | undefined
  const currentModelId = trainResult?.model_id as string | undefined

  const [runs, setRuns] = useState<Experiment[]>([])

  useEffect(() => {
    if (!datasetId || !targetCol) return
    getExperiments({ dataset_id: datasetId, target_col: targetCol, limit: 10 })
      .then(setRuns)
      .catch(() => {})
  }, [datasetId, targetCol, currentModelId])

  if (!trainResult || runs.length < 2) return null

  const taskType = trainResult.task_type as string | undefined
  const primaryMetric = taskType === 'classification' ? 'accuracy' : 'wmape'
  const higherIsBetter = taskType === 'classification'

  const sorted = [...runs].sort((a, b) => {
    const va = a.metrics[primaryMetric] as number | undefined ?? (higherIsBetter ? -1 : 99)
    const vb = b.metrics[primaryMetric] as number | undefined ?? (higherIsBetter ? -1 : 99)
    return higherIsBetter ? vb - va : va - vb
  })

  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">
        All runs for <span className="font-mono text-indigo-700">{targetCol}</span>{' '}
        <span className="text-slate-400 font-normal">({runs.length})</span>
      </h3>
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Model</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">
                {primaryMetric.toUpperCase()}
              </th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">CV mean</th>
              <th className="text-left px-3 py-2 text-slate-500 font-medium w-24">Date</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((run, i) => {
              const isCurrent = run.model_id === currentModelId
              const isBest = i === 0
              const metricVal = run.metrics[primaryMetric] as number | undefined
              const cvMean = run.metrics.cv_mean as number | null | undefined
              return (
                <tr
                  key={run.run_id}
                  className={`border-b border-slate-50 last:border-0 ${isCurrent ? 'bg-indigo-50' : 'hover:bg-slate-50'}`}
                >
                  <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[120px]">
                    {run.model_type}
                    {isCurrent && (
                      <span className="ml-1.5 text-indigo-600 font-sans font-medium">(current)</span>
                    )}
                    {isBest && !isCurrent && (
                      <span className="ml-1.5 text-emerald-600 font-sans font-medium">(best)</span>
                    )}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${isBest ? 'font-semibold text-emerald-700' : 'text-slate-700'}`}>
                    {metricVal != null ? metricVal.toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-500">
                    {cvMean != null ? Math.abs(cvMean).toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-1.5 text-slate-400">
                    {run.created_at.slice(0, 10)}
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
                <MLStoredEvalSummary results={mlResults} />
                <MLTrainSummary results={mlResults} />
                <MLComparisonPanel results={mlResults} datasetId={response?.dataset_id ?? null} />
                <MLExplainSummary results={mlResults} />
                <MLPDPSummary results={mlResults} />
                <MLExplainPredictionSummary results={mlResults} />
                <MLScoreSummary results={mlResults} />
                <MLForecastSummary results={mlResults} />
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
