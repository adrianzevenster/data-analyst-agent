import { useMemo } from 'react'
import { Download } from 'lucide-react'
import clsx from 'clsx'

interface DataTableProps {
  title: string
  columns: string[]
  data: Record<string, unknown>[]
  showDownload?: boolean
}

function isNumeric(val: unknown): boolean {
  if (val === null || val === undefined || val === '') return false
  return typeof val === 'number' || (typeof val === 'string' && !isNaN(Number(val)))
}

function formatCell(val: unknown): string {
  if (val === null || val === undefined) return ''
  if (typeof val === 'number') {
    return Number.isInteger(val) ? val.toLocaleString() : val.toLocaleString(undefined, { maximumFractionDigits: 4 })
  }
  return String(val)
}

function downloadCsv(title: string, columns: string[], data: Record<string, unknown>[]) {
  const header = columns.join(',')
  const rows = data.map((row) =>
    columns.map((c) => {
      const v = String(row[c] ?? '')
      return v.includes(',') || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v
    }).join(',')
  )
  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${title.replace(/\s+/g, '_')}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export default function DataTable({ title, columns, data, showDownload = true }: DataTableProps) {
  const numericCols = useMemo(
    () => new Set(columns.filter((c) => data.slice(0, 10).some((row) => isNumeric(row[c])))),
    [columns, data]
  )

  if (!data.length) return null

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
        <h3 className="text-slate-800 font-semibold text-sm">{title}</h3>
        <div className="flex items-center gap-3">
          <span className="text-slate-400 text-xs">{data.length.toLocaleString()} rows</span>
          {showDownload && (
            <button
              onClick={() => downloadCsv(title, columns, data)}
              className="text-slate-400 hover:text-indigo-600 transition-colors"
              title="Download CSV"
            >
              <Download size={14} />
            </button>
          )}
        </div>
      </div>
      <div className="overflow-auto max-h-72 thin-scroll">
        <table className="w-full text-xs">
          <thead className="sticky top-0 z-10">
            <tr className="bg-slate-50 border-b border-slate-200">
              {columns.map((col) => (
                <th
                  key={col}
                  className={clsx(
                    'px-3 py-2 font-semibold text-slate-600 whitespace-nowrap',
                    numericCols.has(col) ? 'text-right' : 'text-left'
                  )}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={i}
                className={clsx(
                  'border-b border-slate-50 hover:bg-slate-50/80 transition-colors',
                  i % 2 === 0 ? 'bg-white' : 'bg-slate-50/40'
                )}
              >
                {columns.map((col) => {
                  const val = row[col]
                  const numeric = numericCols.has(col)
                  const text = formatCell(val)
                  return (
                    <td
                      key={col}
                      className={clsx(
                        'px-3 py-2 text-slate-700 max-w-[200px] truncate',
                        numeric ? 'text-right font-mono' : 'text-left'
                      )}
                      title={text}
                    >
                      {text}
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
