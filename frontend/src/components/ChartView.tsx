import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import type { ChartSpec } from '../types/api'

const COLORS = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

interface ChartViewProps {
  chart: ChartSpec
}

function ChartCard({ title, caption, children }: { title: string; caption?: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
      <h3 className="text-slate-800 font-semibold text-sm mb-3">{title}</h3>
      {children}
      {caption && <p className="text-slate-400 text-xs mt-2 text-center">{caption}</p>}
    </div>
  )
}

function BarChartView({ chart }: { chart: ChartSpec }) {
  const series = chart.y_series?.length ? chart.y_series : chart.y ? [chart.y] : []
  const isMulti = series.length > 1

  return (
    <ChartCard title={chart.title}>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chart.data} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey={chart.x}
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            tickLine={false}
            angle={-30}
            textAnchor="end"
            interval={0}
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={50} />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
          />
          {isMulti && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {series.map((key, idx) => (
            <Bar key={key} dataKey={key} fill={COLORS[idx % COLORS.length]} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function HistogramView({ chart }: { chart: ChartSpec }) {
  return (
    <ChartCard title={chart.title} caption={chart.x_label ?? chart.column ?? ''}>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chart.data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="bin_label"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            tickLine={false}
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={40} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }} />
          <Bar dataKey="count" fill={COLORS[0]} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function LineChartView({ chart }: { chart: ChartSpec }) {
  const y = chart.y ?? ''
  return (
    <ChartCard title={chart.title}>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chart.data} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey={chart.x}
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            tickLine={false}
            angle={-30}
            textAnchor="end"
            interval={Math.max(0, Math.floor(chart.data.length / 8) - 1)}
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={50} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }} />
          <Line type="monotone" dataKey={y} stroke={COLORS[0]} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

function ScatterChartView({ chart }: { chart: ChartSpec }) {
  const caption = chart.correlation != null ? `Correlation: ${chart.correlation.toFixed(3)}` : undefined
  const x = chart.x
  const y = chart.y ?? ''
  const mapped = chart.data.map((d) => ({ x: d[x], y: d[y] }))

  return (
    <ChartCard title={chart.title} caption={caption}>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="x" name={x} tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} />
          <YAxis dataKey="y" name={y} tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={50} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
          />
          <Scatter data={mapped} fill={COLORS[0]} opacity={0.7} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

export default function ChartView({ chart }: ChartViewProps) {
  if (!chart.data?.length) return null

  switch (chart.type) {
    case 'bar':
      return <BarChartView chart={chart} />
    case 'histogram':
      return <HistogramView chart={chart} />
    case 'line':
      return <LineChartView chart={chart} />
    case 'scatter':
      return <ScatterChartView chart={chart} />
    default:
      return null
  }
}
