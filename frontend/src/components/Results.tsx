import React, { useState, useEffect, useCallback } from 'react'
import { BarChart3, Brain, Target, FlaskConical, Database, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'
import type { ChatResponse, ChartSpec, Experiment, LineageReport, PredictionSetInfo, ToolResult } from '../types/api'
import { getExperiments, startTrainingJob } from '../lib/api'
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

function MLScoreSummary({ results, datasetId }: { results: ToolResult[]; datasetId: string | null }) {
  const [retrainState, setRetrainState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
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
              disabled={retrainState !== 'idle' || !datasetId}
              onClick={async () => {
                if (!datasetId) return
                const target = (scoreResult.target_col as string | undefined) ?? ''
                setRetrainState('loading')
                try {
                  await startTrainingJob({ dataset_id: datasetId, target_col: target, model_type: 'auto' })
                  setRetrainState('done')
                } catch {
                  setRetrainState('error')
                }
              }}
              className="text-amber-700 hover:text-amber-900 underline transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {retrainState === 'loading' ? '…' : retrainState === 'done' ? '✓ Job submitted' : retrainState === 'error' ? '✗ Failed' : 'Retrain now'}
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

// ─── Data tab components ──────────────────────────────────────────────────

function FindingsList({ findings }: { findings: string[] | undefined }) {
  if (!findings || findings.length === 0) return null
  return (
    <div className="flex flex-col gap-1 mb-3">
      {findings.map((f, i) => (
        <div key={i} className="bg-amber-50 border border-amber-200 text-amber-800 text-xs rounded-lg px-3 py-1.5">
          {f}
        </div>
      ))}
    </div>
  )
}

interface ProfileCol {
  name: string
  dtype: string
  missing_pct: number
  unique: number
  mean?: number
  std?: number
  min?: number | string
  max?: number | string
  top_value?: string
  [key: string]: unknown
}

function ProfileColumnTable({ columns }: { columns: ProfileCol[] }) {
  if (!columns || columns.length === 0) return null
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-100">
            <th className="text-left px-3 py-2 text-slate-500 font-medium">Column</th>
            <th className="text-left px-3 py-2 text-slate-500 font-medium">Type</th>
            <th className="text-right px-3 py-2 text-slate-500 font-medium">Missing%</th>
            <th className="text-right px-3 py-2 text-slate-500 font-medium">Unique</th>
            <th className="text-left px-3 py-2 text-slate-500 font-medium">Summary</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((col) => {
            const mp = col.missing_pct ?? 0
            const mpColor = mp === 0 ? 'text-emerald-600' : mp >= 50 ? 'text-rose-600' : 'text-amber-600'
            let summary = '—'
            if (col.mean != null) {
              summary = `μ ${col.mean.toFixed(2)}`
              if (col.std != null) summary += `  σ ${col.std.toFixed(2)}`
            } else if (col.top_value != null) {
              summary = `top: ${col.top_value}`
            } else if (col.min != null && col.max != null) {
              summary = `${String(col.min).slice(0, 10)} → ${String(col.max).slice(0, 10)}`
            }
            return (
              <tr key={col.name} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[120px]">{col.name}</td>
                <td className="px-3 py-1.5">
                  <span className="bg-slate-100 text-slate-600 rounded px-1.5 py-0.5 font-mono">{col.dtype}</span>
                </td>
                <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${mpColor}`}>
                  {mp.toFixed(1)}%
                </td>
                <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-500">
                  {col.unique.toLocaleString()}
                </td>
                <td className="px-3 py-1.5 text-slate-600 truncate max-w-[160px]">{summary}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function InlineCharts({ charts }: { charts: unknown[] | undefined }) {
  if (!charts || charts.length === 0) return null
  const valid = charts.filter((c): c is ChartSpec =>
    typeof c === 'object' && c !== null && 'type' in c && 'data' in c
  )
  if (valid.length === 0) return null
  return (
    <div className={`grid gap-3 mb-3 ${valid.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
      {valid.slice(0, 6).map((chart, i) => <ChartView key={i} chart={chart} />)}
    </div>
  )
}

function ProfileSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'profile_dataset' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Dataset Profile</h3>
      {r.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows" value={Number(r.n_rows ?? 0).toLocaleString()} />
        <MetricCard label="Columns" value={Number(r.n_cols ?? 0)} />
      </div>
      <FindingsList findings={r.findings as string[] | undefined} />
      <ProfileColumnTable columns={(r.columns as ProfileCol[] | undefined) ?? []} />
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function QualitySummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'data_quality_report' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const findings = r.findings as string[] | undefined
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Data Quality</h3>
      {r.engineering_readout != null && (
        <div className={`rounded-xl px-3.5 py-2.5 text-sm mb-3 border ${
          findings && findings.length > 0
            ? 'bg-amber-50 border-amber-200 text-amber-800'
            : 'bg-green-50 border-green-200 text-green-800'
        }`}>
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows" value={Number(r.n_rows ?? 0).toLocaleString()} />
        <MetricCard label="Issues" value={findings?.length ?? 0} />
      </div>
      <FindingsList findings={findings} />
      <ProfileColumnTable columns={(r.columns as ProfileCol[] | undefined) ?? []} />
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function ClusterSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'kmeans_clusters' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const summary = r.cluster_summary as Array<{ cluster_id: number; size: number; size_pct: number }> | undefined
  const sil = r.silhouette_score as number | null | undefined
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">KMeans Clusters</h3>
      {r.engineering_readout != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 text-sm text-indigo-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="k clusters" value={Number(r.k_used ?? r.k_requested ?? 0)} />
        <MetricCard label="Silhouette" value={sil != null ? sil.toFixed(3) : 'n/a'} />
        <MetricCard label="Rows clustered" value={Number(r.n_rows_clustered ?? 0).toLocaleString()} />
      </div>
      {summary && summary.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Cluster</th>
                <th className="text-right px-3 py-2 text-slate-500 font-medium">Size</th>
                <th className="text-left px-3 py-2 text-slate-500 font-medium w-36">Share</th>
              </tr>
            </thead>
            <tbody>
              {summary.map(c => (
                <tr key={c.cluster_id} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                  <td className="px-3 py-1.5 font-mono text-slate-800">Cluster {c.cluster_id}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-700">{c.size.toLocaleString()}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-100 rounded-full h-1.5">
                        <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${c.size_pct}%` }} />
                      </div>
                      <span className="text-slate-500 font-mono w-10 text-right tabular-nums">{c.size_pct.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function AnomalySummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'anomaly_scan' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const topAnomalies = r.top_anomalies as Array<Record<string, unknown>> | undefined
  const nAnomalies = Number(r.n_anomalies ?? 0)
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Anomaly Scan</h3>
      {r.engineering_readout != null && (
        <div className={`rounded-xl px-3.5 py-2.5 text-sm mb-3 border ${
          nAnomalies > 0
            ? 'bg-rose-50 border-rose-200 text-rose-800'
            : 'bg-green-50 border-green-200 text-green-800'
        }`}>
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows scanned" value={Number(r.n_rows_scanned ?? 0).toLocaleString()} />
        <MetricCard label="Anomalies" value={nAnomalies.toLocaleString()} />
        <MetricCard label="Anomaly rate" value={`${Number(r.anomaly_rate_pct ?? 0).toFixed(2)}%`} />
      </div>
      {topAnomalies && topAnomalies.length > 0 && (() => {
        const cols = Object.keys(topAnomalies[0]).filter(k => k !== 'is_anomaly')
        return (
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
            <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide px-3 py-1.5 border-b border-slate-100">
              Top anomalies (most severe first)
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-100">
                    {cols.map(c => <th key={c} className="text-right px-3 py-2 text-slate-500 font-medium whitespace-nowrap">{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {topAnomalies.slice(0, 10).map((row, i) => (
                    <tr key={i} className="border-b border-slate-50 last:border-0 hover:bg-rose-50/30">
                      {cols.map(c => (
                        <td key={c} className="px-3 py-1.5 text-right font-mono text-slate-700 whitespace-nowrap">
                          {row[c] != null
                            ? typeof row[c] === 'number' ? (row[c] as number).toFixed(3) : String(row[c])
                            : '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {topAnomalies.length > 10 && (
              <p className="px-3 py-1.5 text-slate-400 text-xs">…and {topAnomalies.length - 10} more</p>
            )}
          </div>
        )
      })()}
    </div>
  )
}

function CorrelationSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'correlation_analysis' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const pairs = r.numeric_correlations as Array<{ column_a: string; column_b: string; correlation: number; abs_correlation: number }> | undefined
  const findings = r.findings as string[] | undefined
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Correlations</h3>
      {r.engineering_readout != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 text-sm text-indigo-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <FindingsList findings={findings} />
      {pairs && pairs.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Feature A</th>
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Feature B</th>
                <th className="text-left px-3 py-2 text-slate-500 font-medium">Strength</th>
                <th className="text-right px-3 py-2 text-slate-500 font-medium w-16">r</th>
              </tr>
            </thead>
            <tbody>
              {pairs.slice(0, 15).map((p, i) => {
                const positive = p.correlation >= 0
                const pct = Math.abs(p.correlation) * 100
                const strong = Math.abs(p.correlation) >= 0.7
                return (
                  <tr key={i} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                    <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[100px]">{p.column_a}</td>
                    <td className="px-3 py-1.5 font-mono text-slate-800 truncate max-w-[100px]">{p.column_b}</td>
                    <td className="px-3 py-1.5">
                      <div className="bg-slate-100 rounded-full h-1.5 min-w-[60px]">
                        <div
                          className={`${positive ? 'bg-indigo-500' : 'bg-rose-400'} h-1.5 rounded-full`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </td>
                    <td className={`px-3 py-1.5 text-right font-mono tabular-nums font-semibold ${
                      strong ? (positive ? 'text-indigo-700' : 'text-rose-700') : 'text-slate-600'
                    }`}>
                      {p.correlation >= 0 ? '+' : ''}{p.correlation.toFixed(3)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function TrendSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'trend_analysis' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const direction = r.direction as string | undefined
  const overallChange = r.overall_change_pct as number | null | undefined
  const directionIcon = direction === 'up' ? '↑' : direction === 'down' ? '↓' : '→'
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Trend Analysis</h3>
      {r.engineering_readout != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 text-sm text-indigo-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {direction && <MetricCard label="Direction" value={`${directionIcon} ${direction}`} />}
        {overallChange != null && (
          <MetricCard label="Overall change" value={`${overallChange >= 0 ? '+' : ''}${overallChange.toFixed(1)}%`} />
        )}
        {r.latest_value != null && (
          <MetricCard label="Latest value" value={Number(r.latest_value).toLocaleString()} />
        )}
        <MetricCard label="Periods" value={Number(r.n_periods ?? 0)} />
      </div>
      {r.peak_period != null && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-3 py-2 mb-3 grid grid-cols-2 gap-3 text-xs">
          <div>
            <p className="text-slate-400 mb-0.5">Peak: <span className="text-slate-600 font-mono">{String(r.peak_period)}</span></p>
            <p className="text-slate-800 font-mono font-semibold">{Number(r.peak_value).toLocaleString()}</p>
          </div>
          <div>
            <p className="text-slate-400 mb-0.5">Trough: <span className="text-slate-600 font-mono">{String(r.trough_period)}</span></p>
            <p className="text-slate-800 font-mono font-semibold">{Number(r.trough_value).toLocaleString()}</p>
          </div>
        </div>
      )}
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function AutoInsightsSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'auto_insights' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const insights = r.insights as Array<{ rank: number; finding: string }> | undefined
  const analysesRun = r.analyses_run as string[] | undefined
  const analysisBadge: Record<string, string> = {
    data_quality:  'bg-amber-50 text-amber-700 border-amber-200',
    relationships: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    anomalies:     'bg-rose-50 text-rose-700 border-rose-200',
    trend:         'bg-teal-50 text-teal-700 border-teal-200',
  }
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Auto Insights</h3>
      {r.engineering_readout != null && (
        <div className="bg-green-50 border border-green-200 rounded-xl px-3.5 py-2.5 text-sm text-green-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricCard label="Rows" value={Number(r.n_rows ?? 0).toLocaleString()} />
        <MetricCard label="Findings" value={insights?.length ?? 0} />
      </div>
      {analysesRun && analysesRun.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {analysesRun.map(a => (
            <span key={a} className={`text-xs px-2 py-0.5 rounded border font-medium ${analysisBadge[a] ?? 'bg-slate-50 text-slate-600 border-slate-200'}`}>
              {a.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}
      {insights && insights.length > 0 && (
        <div className="space-y-1.5 mb-3">
          {insights.map(item => (
            <div key={item.rank} className="flex items-start gap-2.5 bg-white border border-slate-200 rounded-xl px-3 py-2">
              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center mt-0.5">
                {item.rank}
              </span>
              <p className="text-slate-700 text-xs leading-relaxed">{item.finding}</p>
            </div>
          ))}
        </div>
      )}
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function SkewedFeaturesSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'skewed_features' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  const features = r.features as Array<{ column: string; skewness: number; severity: string }> | undefined
  if (!features || features.length === 0) return null
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Skewed Features</h3>
      {r.engineering_readout != null && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl px-3.5 py-2.5 text-sm text-amber-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-3">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Column</th>
              <th className="text-right px-3 py-2 text-slate-500 font-medium">Skewness</th>
              <th className="text-left px-3 py-2 text-slate-500 font-medium">Severity</th>
            </tr>
          </thead>
          <tbody>
            {features.map(f => (
              <tr key={f.column} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
                <td className="px-3 py-1.5 font-mono text-slate-800">{f.column}</td>
                <td className="px-3 py-1.5 text-right font-mono tabular-nums text-slate-700">{f.skewness.toFixed(3)}</td>
                <td className="px-3 py-1.5">
                  <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                    f.severity === 'high' ? 'bg-rose-50 text-rose-700' : 'bg-amber-50 text-amber-700'
                  }`}>{f.severity}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

function OverrepresentedSummary({ results }: { results: ToolResult[] }) {
  const r = results.find(t => t.name === 'overrepresented_categories' && t.ok)?.result as Record<string, unknown> | undefined
  if (!r || 'error' in r) return null
  return (
    <div>
      <h3 className="text-slate-700 font-semibold text-sm mb-2">Category Distribution</h3>
      {r.engineering_readout != null && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3.5 py-2.5 text-sm text-indigo-800 mb-3">
          {String(r.engineering_readout)}
        </div>
      )}
      <InlineCharts charts={r.charts as unknown[] | undefined} />
    </div>
  )
}

// ─── Experiment tab helpers ────────────────────────────────────────────────

function formatAgo(date: Date): string {
  const secs = Math.floor((Date.now() - date.getTime()) / 1000)
  if (secs < 10) return 'just now'
  if (secs < 60) return `${secs}s ago`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m ago`
  return `${Math.floor(mins / 60)}h ago`
}

function MetricTrend({
  values,
  metric,
  higherIsBetter,
  gradientId,
}: {
  values: (number | undefined)[]
  metric: string
  higherIsBetter: boolean
  gradientId: string
}) {
  const valid = values.filter((v): v is number => v != null)
  if (valid.length < 2) return null

  const min = Math.min(...valid)
  const max = Math.max(...valid)
  const range = max - min

  const W = 300, H = 44, PX = 4, PY = 5
  const plotW = W - 2 * PX
  const plotH = H - 2 * PY

  const coords: Array<[number, number] | null> = values.map((v, i) => {
    if (v == null) return null
    const x = PX + (values.length > 1 ? (i / (values.length - 1)) * plotW : plotW / 2)
    const y = range === 0 ? PY + plotH / 2 : H - PY - ((v - min) / range) * plotH
    return [x, y]
  })

  let d = ''
  coords.forEach((c, i) => {
    if (!c) return
    d += i === 0 || !coords[i - 1] ? `M ${c[0]} ${c[1]} ` : `L ${c[0]} ${c[1]} `
  })

  const firstCoord = coords.find(Boolean)
  const lastCoord = [...coords].reverse().find(Boolean)
  const delta = valid[valid.length - 1] - valid[0]
  const improved = higherIsBetter ? delta >= 0 : delta <= 0
  const trendColor = improved ? '#10b981' : '#ef4444'

  return (
    <div className="bg-white border border-slate-200 rounded-xl px-4 py-3 mb-3">
      <div className="flex items-center justify-between mb-1">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide">
          {metric} trend · {values.length} runs
        </p>
        <span className="text-xs font-mono font-semibold" style={{ color: trendColor }}>
          {delta >= 0 ? '+' : ''}{delta.toFixed(4)}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-slate-400 text-xs font-mono w-12 text-right flex-shrink-0">{min.toFixed(3)}</span>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="flex-1 h-11">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
            </linearGradient>
          </defs>
          {firstCoord && lastCoord && (
            <path
              d={`${d} L ${lastCoord[0]} ${H} L ${firstCoord[0]} ${H} Z`}
              fill={`url(#${gradientId})`}
            />
          )}
          <path d={d} fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          {coords.map((c, i) => c ? <circle key={i} cx={c[0]} cy={c[1]} r="2.5" fill="#6366f1" /> : null)}
        </svg>
        <span className="text-slate-400 text-xs font-mono w-12 flex-shrink-0">{max.toFixed(3)}</span>
      </div>
    </div>
  )
}

function RunBadges({ run }: { run: Experiment }) {
  const addInteractions = run.preprocessing.interaction_features_added as boolean | undefined
  const lagCols = run.preprocessing.lag_feature_cols as string[] | undefined
  const textCols = run.preprocessing.text_feature_cols as string[] | undefined
  const datetimeCols = run.preprocessing.datetime_feature_cols as string[] | undefined
  const calibrated = run.metrics.calibrated as boolean | undefined
  const tune = run.params.tune as boolean | undefined
  const imbalanceRatio = run.metrics.imbalance_ratio as number | null | undefined
  const smote = imbalanceRatio != null && imbalanceRatio > 5

  const badges: Array<{ label: string; cls: string; title?: string }> = []
  if (addInteractions) badges.push({ label: '↔ interactions', cls: 'bg-cyan-50 text-cyan-700 border-cyan-200', title: 'Degree-2 pairwise interaction features' })
  if (lagCols && lagCols.length > 0) badges.push({ label: `⏱ lag ×${lagCols.length}`, cls: 'bg-teal-50 text-teal-700 border-teal-200', title: lagCols.join(', ') })
  if (textCols && textCols.length > 0) badges.push({ label: `T text ×${textCols.length}`, cls: 'bg-purple-50 text-purple-700 border-purple-200', title: textCols.join(', ') })
  if (datetimeCols && datetimeCols.length > 0) badges.push({ label: '📅 datetime', cls: 'bg-blue-50 text-blue-700 border-blue-200' })
  if (calibrated) badges.push({ label: 'Platt calibrated', cls: 'bg-violet-50 text-violet-700 border-violet-200' })
  if (tune) badges.push({ label: 'HPO', cls: 'bg-amber-50 text-amber-700 border-amber-200' })
  if (smote) badges.push({ label: `SMOTE ${imbalanceRatio?.toFixed(1)}×`, cls: 'bg-rose-50 text-rose-700 border-rose-200', title: 'Synthetic minority oversampling applied' })

  if (badges.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1">
      {badges.map((b, i) => (
        <span key={i} title={b.title} className={`text-xs px-1.5 py-0.5 rounded border font-medium ${b.cls}`}>
          {b.label}
        </span>
      ))}
    </div>
  )
}

function ExperimentRunCard({
  run,
  primaryMetric,
  isBest,
  expanded,
  onToggle,
}: {
  run: Experiment
  primaryMetric: string
  isBest: boolean
  expanded: boolean
  onToggle: () => void
}) {
  const metricVal = run.metrics[primaryMetric] as number | undefined
  const cvMean = run.metrics.cv_mean as number | null | undefined
  const cvStd = run.metrics.cv_std as number | null | undefined
  const bestParams = run.params.best_params as Record<string, unknown> | null | undefined
  const autoDroppedIdCols = run.preprocessing.auto_dropped_id_cols as string[] | undefined
  const comparison = run.comparison as {
    metric?: string; previous?: number; current?: number; delta?: number
    improved?: boolean; previous_model_type?: string
  } | null | undefined

  const numericMetrics = (Object.entries(run.metrics) as [string, unknown][]).filter(
    ([k, v]) => typeof v === 'number' && k !== 'imbalance_ratio'
  ) as [string, number][]

  return (
    <div className={`bg-white rounded-xl border shadow-sm ${isBest ? 'border-emerald-200' : 'border-slate-200'}`}>
      <button onClick={onToggle} className="w-full text-left px-3.5 py-2.5">
        <div className="flex items-start gap-2 flex-wrap">
          <div className="flex items-center gap-2 min-w-0 flex-shrink-0">
            <span className="font-mono text-slate-800 text-xs font-semibold">{run.model_type}</span>
            {isBest && (
              <span className="text-xs px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">best</span>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <RunBadges run={run} />
          </div>
          <div className="flex items-center gap-3 flex-shrink-0 ml-auto">
            <span className={`font-mono tabular-nums text-xs font-semibold ${isBest ? 'text-emerald-700' : 'text-slate-700'}`}>
              {primaryMetric.toUpperCase()} {metricVal != null ? metricVal.toFixed(4) : '—'}
            </span>
            {cvMean != null && (
              <span className="text-slate-400 text-xs font-mono tabular-nums">
                CV {Math.abs(cvMean).toFixed(3)}{cvStd != null ? ` ±${cvStd.toFixed(3)}` : ''}
              </span>
            )}
            <span className="text-slate-400 text-xs">{run.created_at.slice(0, 10)}</span>
            {expanded
              ? <ChevronDown size={14} className="text-slate-400" />
              : <ChevronRight size={14} className="text-slate-400" />}
          </div>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-slate-100 px-3.5 py-3 space-y-3">
          {numericMetrics.length > 0 && (
            <div>
              <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide mb-1.5">All metrics</p>
              <div className="grid grid-cols-3 gap-1.5">
                {numericMetrics.map(([k, v]) => (
                  <div key={k} className="bg-slate-50 rounded-lg px-2.5 py-1.5">
                    <p className="text-slate-400 text-xs leading-tight">{k.replace(/_/g, ' ')}</p>
                    <p className="text-slate-800 text-sm font-mono font-semibold leading-tight mt-0.5">{v.toFixed(4)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {comparison != null && comparison.delta != null && (
            <div className={`rounded-lg px-3 py-2 text-xs border ${
              comparison.improved
                ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                : 'bg-rose-50 border-rose-200 text-rose-800'
            }`}>
              vs {comparison.previous_model_type ?? 'previous'}:{' '}
              {comparison.metric?.toUpperCase()} {comparison.previous?.toFixed(4)} → {comparison.current?.toFixed(4)}{' '}
              <span className="font-semibold">
                ({comparison.improved ? '↑' : '↓'}{Math.abs(comparison.delta).toFixed(4)})
              </span>
            </div>
          )}

          {bestParams != null && Object.keys(bestParams).length > 0 && (
            <div>
              <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide mb-1.5">HPO best params</p>
              <div className="bg-white rounded-lg border border-slate-200 divide-y divide-slate-50">
                {Object.entries(bestParams).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs px-3 py-1 first:pt-1.5 last:pb-1.5">
                    <span className="text-slate-500 font-mono">{k}</span>
                    <span className="text-slate-800 font-mono font-medium">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {autoDroppedIdCols && autoDroppedIdCols.length > 0 && (
            <div>
              <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide mb-1.5">
                Auto-dropped ID columns ({autoDroppedIdCols.length})
              </p>
              <div className="flex flex-wrap gap-1">
                {autoDroppedIdCols.map(c => (
                  <span key={c} className="font-mono text-xs bg-slate-100 text-slate-600 rounded px-1.5 py-0.5">{c}</span>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-1 border-t border-slate-50">
            <span className="text-slate-400 text-xs">Model ID</span>
            <span className="text-slate-500 font-mono text-xs">{run.model_id.slice(0, 8)}…</span>
          </div>
        </div>
      )}
    </div>
  )
}

function ExperimentGroup({
  targetCol,
  runs,
  expandedRun,
  onToggleExpand,
}: {
  targetCol: string
  runs: Experiment[]
  expandedRun: string | null
  onToggleExpand: (id: string) => void
}) {
  const taskType = runs[0]?.task_type ?? 'unknown'
  const primaryMetric = taskType === 'classification' ? 'accuracy' : 'wmape'
  const higherIsBetter = taskType === 'classification'

  const bestRun = [...runs].sort((a, b) => {
    const va = (a.metrics[primaryMetric] as number | undefined) ?? (higherIsBetter ? -Infinity : Infinity)
    const vb = (b.metrics[primaryMetric] as number | undefined) ?? (higherIsBetter ? -Infinity : Infinity)
    return higherIsBetter ? vb - va : va - vb
  })[0]

  // API returns DESC; reverse to chronological for sparkline
  const chronoValues = [...runs].reverse().map(r => r.metrics[primaryMetric] as number | undefined)
  const gradientId = `trendFill-${targetCol.replace(/[^a-zA-Z0-9]/g, '_')}`

  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-slate-700 font-semibold text-sm">
          Target: <span className="font-mono text-indigo-700">{targetCol}</span>
        </h3>
        <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 font-medium">{taskType}</span>
        <span className="text-xs text-slate-400">{runs.length} run{runs.length !== 1 ? 's' : ''}</span>
      </div>
      <MetricTrend
        values={chronoValues}
        metric={primaryMetric}
        higherIsBetter={higherIsBetter}
        gradientId={gradientId}
      />
      <div className="space-y-2">
        {runs.map(run => (
          <ExperimentRunCard
            key={run.run_id}
            run={run}
            primaryMetric={primaryMetric}
            isBest={run.run_id === bestRun?.run_id}
            expanded={expandedRun === run.run_id}
            onToggle={() => onToggleExpand(run.run_id)}
          />
        ))}
      </div>
    </div>
  )
}

function ExperimentsTab({ datasetId, refreshTick }: { datasetId: string | null; refreshTick: number }) {
  const [runs, setRuns] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(false)
  const [lastFetched, setLastFetched] = useState<Date | null>(null)
  const [expandedRun, setExpandedRun] = useState<string | null>(null)

  const fetchRuns = useCallback(() => {
    setLoading(true)
    getExperiments({ dataset_id: datasetId ?? undefined, limit: 100 })
      .then(data => { setRuns(data); setLastFetched(new Date()) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [datasetId])

  useEffect(() => { fetchRuns() }, [fetchRuns, refreshTick])

  const groups = new Map<string, Experiment[]>()
  for (const run of runs) {
    if (!groups.has(run.target_col)) groups.set(run.target_col, [])
    groups.get(run.target_col)!.push(run)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-slate-500 text-xs">
          {runs.length > 0
            ? `${runs.length} run${runs.length !== 1 ? 's' : ''}${lastFetched ? ` · ${formatAgo(lastFetched)}` : ''}`
            : 'No runs loaded'}
        </p>
        <button
          onClick={fetchRuns}
          disabled={loading}
          className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 disabled:opacity-50 transition-colors"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {loading && runs.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <p className="text-slate-400 text-sm">Loading experiments…</p>
        </div>
      ) : runs.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <p className="text-slate-400 text-sm text-center">
            No experiments yet.
            <br />
            Train a model to populate this tab.
          </p>
        </div>
      ) : (
        Array.from(groups.entries()).map(([targetCol, groupRuns]) => (
          <ExperimentGroup
            key={targetCol}
            targetCol={targetCol}
            runs={groupRuns}
            expandedRun={expandedRun}
            onToggleExpand={(id) => setExpandedRun(prev => prev === id ? null : id)}
          />
        ))
      )}
    </div>
  )
}

// ─── Tab bar ───────────────────────────────────────────────────────────────

type Tab = 'latest' | 'data' | 'model' | 'eval' | 'experiments'

const ML_MODEL_TOOL_NAMES = new Set([
  'train_supervised_model',
  'explain_model',
  'shap_explain_prediction',
  'score_with_model',
  'forecast_with_model',
  'compute_pdp',
])

const ML_EVAL_TOOL_NAMES = new Set([
  'evaluate_ml_predictions',
  'evaluate_trained_model',
])

const DATA_TOOL_NAMES = new Set([
  'profile_dataset',
  'data_quality_report',
  'kmeans_clusters',
  'anomaly_scan',
  'correlation_analysis',
  'trend_analysis',
  'auto_insights',
  'overrepresented_categories',
  'skewed_features',
])

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
  count,
}: {
  active: boolean
  onClick: () => void
  icon: React.ElementType
  label: string
  count?: number
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
        active
          ? 'bg-indigo-50 text-indigo-700'
          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
      }`}
    >
      <Icon size={13} />
      {label}
      {count != null && count > 0 && (
        <span className={`px-1.5 py-0.5 rounded-full text-xs font-semibold leading-none ${
          active ? 'bg-indigo-100 text-indigo-600' : 'bg-slate-100 text-slate-500'
        }`}>
          {count}
        </span>
      )}
    </button>
  )
}

// ─── Main Results component ────────────────────────────────────────────────

const Results = React.memo(function Results({ response, conversationId }: ResultsProps) {
  const [mlResults, setMlResults] = useState<ToolResult[]>([])
  const [dataResults, setDataResults] = useState<ToolResult[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('latest')
  const [experimentsTick, setExperimentsTick] = useState(0)

  useEffect(() => {
    setMlResults([])
    setDataResults([])
    setActiveTab('latest')
  }, [conversationId])

  useEffect(() => {
    if (!response) return

    // Merge ML tool results
    const mlIncoming = response.tool_results.filter(r => ML_TOOL_NAMES.has(r.name) && r.ok)
    if (mlIncoming.length > 0) {
      setMlResults(prev => {
        const map = new Map(prev.map(r => [r.name, r]))
        for (const r of mlIncoming) map.set(r.name, r)
        return Array.from(map.values())
      })
    }

    // Merge data tool results
    const dataIncoming = response.tool_results.filter(r => DATA_TOOL_NAMES.has(r.name) && r.ok)
    if (dataIncoming.length > 0) {
      setDataResults(prev => {
        const map = new Map(prev.map(r => [r.name, r]))
        for (const r of dataIncoming) map.set(r.name, r)
        return Array.from(map.values())
      })
    }

    // Auto-switch to the most relevant tab
    const okNames = new Set(response.tool_results.filter(r => r.ok).map(r => r.name))
    if ([...okNames].some(n => ML_EVAL_TOOL_NAMES.has(n))) {
      setActiveTab('eval')
    } else if ([...okNames].some(n => ML_MODEL_TOOL_NAMES.has(n))) {
      setActiveTab('model')
      if (okNames.has('train_supervised_model')) setExperimentsTick(t => t + 1)
    } else if ([...okNames].some(n => DATA_TOOL_NAMES.has(n))) {
      setActiveTab('data')
    } else if (response.tables.length > 0 || response.charts.length > 0) {
      setActiveTab('latest')
    }
  }, [response])

  const modelResults = mlResults.filter(r => ML_MODEL_TOOL_NAMES.has(r.name))
  const evalResults = mlResults.filter(r => ML_EVAL_TOOL_NAMES.has(r.name))
  const latestCount = (response?.tables.length ?? 0) + (response?.charts.length ?? 0)

  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-slate-50">
      {/* Tab bar */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-slate-200 bg-white flex-wrap">
        <TabButton
          active={activeTab === 'latest'} onClick={() => setActiveTab('latest')}
          icon={BarChart3} label="Latest" count={latestCount}
        />
        <TabButton
          active={activeTab === 'data'} onClick={() => setActiveTab('data')}
          icon={Database} label="Data" count={dataResults.length || undefined}
        />
        <TabButton
          active={activeTab === 'model'} onClick={() => setActiveTab('model')}
          icon={Brain} label="Model" count={modelResults.length || undefined}
        />
        <TabButton
          active={activeTab === 'eval'} onClick={() => setActiveTab('eval')}
          icon={Target} label="Eval" count={evalResults.length || undefined}
        />
        <TabButton
          active={activeTab === 'experiments'} onClick={() => setActiveTab('experiments')}
          icon={FlaskConical} label="Experiments"
        />
      </div>

      <div className="flex-1 overflow-y-auto thin-scroll px-4 py-4 space-y-4">
        {/* Latest tab — tables and charts from the most recent query */}
        {activeTab === 'latest' && (
          latestCount === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-400 text-sm text-center">
                Upload a dataset and run a query to see results.
              </p>
            </div>
          ) : (
            <>
              {response?.tables.map((table, i) => (
                <DataTable key={`${table.title}-${i}`} title={table.title} columns={table.columns} data={table.data} />
              ))}
              {response?.charts.map((chart, i) => (
                <ChartView key={`${chart.title}-${i}`} chart={chart} />
              ))}
            </>
          )
        )}

        {/* Data tab — profile, quality, clusters, anomalies, correlations, trends */}
        {activeTab === 'data' && (
          dataResults.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-400 text-sm text-center">
                No data analysis yet.
                <br />
                Ask to profile, explore, cluster, or find anomalies.
              </p>
            </div>
          ) : (
            <>
              <ProfileSummary results={dataResults} />
              <QualitySummary results={dataResults} />
              <CorrelationSummary results={dataResults} />
              <TrendSummary results={dataResults} />
              <ClusterSummary results={dataResults} />
              <AnomalySummary results={dataResults} />
              <AutoInsightsSummary results={dataResults} />
              <SkewedFeaturesSummary results={dataResults} />
              <OverrepresentedSummary results={dataResults} />
            </>
          )
        )}

        {/* Model tab — train, score, explain, forecast, PDP */}
        {activeTab === 'model' && (
          modelResults.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-400 text-sm text-center">
                No model results yet.
                <br />
                Train, score, or explain a model to populate this tab.
              </p>
            </div>
          ) : (
            <>
              <MLTrainSummary results={mlResults} />
              <MLComparisonPanel results={mlResults} datasetId={response?.dataset_id ?? null} />
              <MLExplainSummary results={mlResults} />
              <MLPDPSummary results={mlResults} />
              <MLExplainPredictionSummary results={mlResults} />
              <MLScoreSummary results={mlResults} datasetId={response?.dataset_id ?? null} />
              <MLForecastSummary results={mlResults} />
            </>
          )
        )}

        {/* Eval tab — ROC, PR, confusion matrix, classification report */}
        {activeTab === 'eval' && (
          evalResults.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-400 text-sm text-center">
                No evaluation results yet.
                <br />
                Run evaluate_trained_model or evaluate_ml_predictions.
              </p>
            </div>
          ) : (
            <>
              <MLEvalSummary results={mlResults} />
              <MLStoredEvalSummary results={mlResults} />
            </>
          )
        )}

        {/* Experiments tab — full run history with metadata */}
        {activeTab === 'experiments' && (
          <ExperimentsTab
            datasetId={response?.dataset_id ?? null}
            refreshTick={experimentsTick}
          />
        )}
      </div>
    </div>
  )
})

export default Results
