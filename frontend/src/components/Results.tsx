import React from 'react'
import { LayoutDashboard } from 'lucide-react'
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
            {/* ML summaries */}
            {response.tool_results.length > 0 && (
              <>
                <MLEvalSummary results={response.tool_results} />
                <MLTrainSummary results={response.tool_results} />
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

            {/* No results at all */}
            {response.tables.length === 0 &&
              response.charts.length === 0 &&
              !response.tool_results.some((r) =>
                ['train_supervised_model', 'evaluate_ml_predictions', 'score_with_model'].includes(r.name)
              ) && (
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-4 py-6 text-center">
                  <p className="text-slate-400 text-sm">No visual output for this query.</p>
                  <p className="text-slate-400 text-xs mt-1">
                    See the chat panel for the analysis narrative.
                  </p>
                </div>
              )}
          </>
        )}
      </div>
    </div>
  )
})

export default Results
