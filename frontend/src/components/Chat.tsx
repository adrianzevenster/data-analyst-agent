import { useState, useRef, useEffect, useCallback, useId } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { MessageSquare, Send, Plus, CheckCircle2, XCircle, Loader2, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getHistory } from '../lib/api'
import type { ChatResponse, ConversationTurn, ToolCall, ToolProgress, SSEEvent } from '../types/api'

interface ChatProps {
  datasetId: string | null
  conversationId: string | null
  onConversationChange: (id: string) => void
  onResponse: (r: ChatResponse) => void
  onNewConversation: () => void
}

function Avatar({ role }: { role: 'user' | 'assistant' }) {
  return (
    <div
      className={clsx(
        'w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0',
        role === 'user'
          ? 'bg-slate-600 text-slate-100'
          : 'bg-indigo-600 text-white'
      )}
    >
      {role === 'user' ? 'U' : 'A'}
    </div>
  )
}

function AssistantMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-2">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-2">{children}</ol>,
        li: ({ children }) => <li className="text-slate-700">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        h1: ({ children }) => <h1 className="font-bold text-base mb-1 mt-2">{children}</h1>,
        h2: ({ children }) => <h2 className="font-semibold text-sm mb-1 mt-2">{children}</h2>,
        h3: ({ children }) => <h3 className="font-semibold text-sm mb-1 mt-1">{children}</h3>,
        code: ({ children }) => (
          <code className="bg-slate-100 text-indigo-700 px-1 py-0.5 rounded text-xs font-mono">
            {children}
          </code>
        ),
        pre: ({ children }) => (
          <pre className="bg-slate-100 rounded p-2 overflow-x-auto text-xs font-mono mb-2">
            {children}
          </pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-indigo-300 pl-3 text-slate-600 italic mb-2">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto mb-2">
            <table className="text-xs border-collapse w-full">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-semibold">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-slate-200 px-2 py-1">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

function HistoryJudgePanel({ turn }: { turn: ConversationTurn }) {
  const score = turn.groundedness_score
  return (
    <JudgePanel
      response={{
        llm_enabled: true,
        groundedness_score: score,
        groundedness_criteria: turn.groundedness_criteria ?? {},
        groundedness_issues: turn.groundedness_issues ?? [],
        judge_status: turn.judge_status ?? (score == null ? 'rule_based' : 'judged'),
        synthesis_source: turn.synthesis_source as 'llm' | 'rules',
      } as ChatResponse}
    />
  )
}

function FeedbackButtons({ conversationId, turnIdx }: { conversationId: string | null; turnIdx: number }) {
  const [rating, setRating] = useState<'up' | 'down' | null>(null)
  const [busy, setBusy] = useState(false)

  if (!conversationId) return null

  const vote = async (r: 'up' | 'down') => {
    if (busy || rating) return
    setBusy(true)
    try {
      const { submitFeedback } = await import('../lib/api')
      await submitFeedback({ conversation_id: conversationId, turn_idx: turnIdx, rating: r })
      setRating(r)
    } catch {
      // silently ignore
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="pl-9 flex items-center gap-1 mt-0.5">
      <button
        onClick={() => vote('up')}
        disabled={busy || rating !== null}
        title="Helpful"
        className={clsx(
          'text-xs px-1.5 py-0.5 rounded transition-colors',
          rating === 'up'
            ? 'text-green-600 bg-green-50'
            : 'text-slate-400 hover:text-green-600 hover:bg-green-50',
          (busy || rating !== null) && 'cursor-default'
        )}
      >
        ↑
      </button>
      <button
        onClick={() => vote('down')}
        disabled={busy || rating !== null}
        title="Not helpful"
        className={clsx(
          'text-xs px-1.5 py-0.5 rounded transition-colors',
          rating === 'down'
            ? 'text-red-600 bg-red-50'
            : 'text-slate-400 hover:text-red-600 hover:bg-red-50',
          (busy || rating !== null) && 'cursor-default'
        )}
      >
        ↓
      </button>
      {rating && (
        <span className={clsx('text-xs', rating === 'up' ? 'text-green-600' : 'text-red-500')}>
          {rating === 'up' ? 'Thanks!' : 'Noted'}
        </span>
      )}
    </div>
  )
}

function Message({ turn, conversationId, turnIdx }: { turn: ConversationTurn; conversationId: string | null; turnIdx: number }) {
  const isUser = turn.role === 'user'
  return (
    <div className="space-y-2">
      <div className={clsx('flex gap-2.5', isUser ? 'flex-row-reverse' : 'flex-row')}>
        <Avatar role={turn.role} />
        <div
          className={clsx(
            'max-w-[80%] px-3.5 py-2.5 text-sm leading-relaxed',
            isUser
              ? 'bg-indigo-600 text-white rounded-2xl rounded-tr-sm'
              : 'bg-white border border-slate-200 text-slate-800 rounded-2xl rounded-tl-sm shadow-sm'
          )}
        >
          {isUser ? turn.content : <AssistantMarkdown content={turn.content} />}
        </div>
      </div>
      {!isUser && (
        <div className="space-y-1">
          <div className="pl-9">
            <span
              className={clsx(
                'text-xs px-2 py-0.5 rounded-full font-medium',
                turn.synthesis_source === 'llm'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-slate-100 text-slate-500'
              )}
            >
              {turn.synthesis_source === 'llm' ? 'LLM synthesis' : 'Rule-based'}
            </span>
          </div>
          <HistoryJudgePanel turn={turn} />
          <FeedbackButtons conversationId={conversationId} turnIdx={turnIdx} />
        </div>
      )}
    </div>
  )
}

function ToolProgressList({
  planned,
  progress,
  synthesizing,
}: {
  planned: ToolCall[]
  progress: ToolProgress[]
  synthesizing: boolean
}) {
  if (!planned.length) return null

  // Use a stable key per (name, index) pair so duplicate tool names don't collide.
  const progressByIndex = new Map(progress.map((p, i) => [i, p]))
  // First planned tool not yet in progress is the currently-executing one.
  const completedNames = new Set(progress.map((p) => p.name))
  let foundRunning = false

  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 flex items-center justify-center flex-shrink-0">
        <Loader2 size={16} className="text-indigo-400 animate-spin" />
      </div>
      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm shadow-sm px-3.5 py-2.5 space-y-1">
        {planned.map((tc, idx) => {
          const p = progressByIndex.get(idx) ?? (completedNames.has(tc.name) ? progress.find((x) => x.name === tc.name) : undefined)
          const isRunning = !p && !foundRunning
          if (isRunning) foundRunning = true
          return (
            <div key={`${tc.name}-${idx}`} className="flex items-center gap-2 text-xs">
              {!p ? (
                isRunning ? (
                  <Loader2 size={12} className="text-indigo-400 animate-spin flex-shrink-0" />
                ) : (
                  <div className="w-3 h-3 rounded-full border-2 border-slate-200 flex-shrink-0" />
                )
              ) : p.status === 'ok' ? (
                <CheckCircle2 size={13} className="text-green-500 flex-shrink-0" />
              ) : (
                <XCircle size={13} className="text-red-500 flex-shrink-0" />
              )}
              <span className={clsx(
                'font-mono',
                !p && isRunning ? 'text-indigo-600' :
                !p ? 'text-slate-300' :
                p.status === 'ok' ? 'text-slate-700' : 'text-red-600'
              )}>
                {tc.name}
              </span>
              {p?.error && <span className="text-red-500 truncate max-w-[160px]">{p.error}</span>}
            </div>
          )
        })}
        {synthesizing && (
          <div className="flex items-center gap-2 text-xs pt-0.5 border-t border-slate-100 mt-1">
            <Loader2 size={12} className="text-violet-400 animate-spin flex-shrink-0" />
            <span className="text-violet-500 italic">Synthesizing…</span>
          </div>
        )}
      </div>
    </div>
  )
}

function scoreColor(n: number) {
  if (n >= 4) return { dot: 'bg-green-500', text: 'text-green-700', bar: 'bg-green-400' }
  if (n === 3) return { dot: 'bg-yellow-400', text: 'text-yellow-700', bar: 'bg-yellow-400' }
  return { dot: 'bg-red-400', text: 'text-red-600', bar: 'bg-red-400' }
}

function judgeStatusMeta(response: ChatResponse) {
  const score = response.groundedness_score
  const status = response.judge_status ?? (
    score != null ? 'judged' : response.synthesis_source === 'llm' ? 'not_sampled' : response.llm_enabled ? 'rule_based' : 'llm_disabled'
  )

  if (status === 'judged' && score != null) {
    const c = scoreColor(score)
    return {
      status,
      label: `${score}/5`,
      detail: null,
      iconClass: c.text,
      labelClass: c.text,
    }
  }
  if (status === 'not_sampled') {
    return {
      status,
      label: 'Not sampled',
      detail: 'Skipped by LLM_JUDGE_SAMPLE_RATE.',
      iconClass: 'text-slate-400',
      labelClass: 'text-slate-500',
    }
  }
  if (status === 'llm_disabled') {
    return {
      status,
      label: 'LLM disabled',
      detail: 'Judging only runs for LLM-synthesized replies.',
      iconClass: 'text-slate-400',
      labelClass: 'text-slate-500',
    }
  }
  if (status === 'failed') {
    return {
      status,
      label: 'Failed',
      detail: 'The reply was delivered, but the judge call failed.',
      iconClass: 'text-red-500',
      labelClass: 'text-red-600',
    }
  }
  return {
    status,
    label: 'Rule-based',
    detail: 'Rule-based replies are not sent to the LLM judge.',
    iconClass: 'text-slate-400',
    labelClass: 'text-slate-500',
  }
}

function CriterionRow({ label, score }: { label: string; score: number }) {
  const c = scoreColor(score)
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 text-xs text-slate-500 capitalize">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full', c.bar)} style={{ width: `${(score / 5) * 100}%` }} />
      </div>
      <span className={clsx('text-xs font-semibold w-6 text-right', c.text)}>{score}/5</span>
    </div>
  )
}

function JudgePanel({ response }: { response: ChatResponse }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const score = response.groundedness_score
  const meta = judgeStatusMeta(response)
  const isJudged = meta.status === 'judged' && score != null
  const criteria = response.groundedness_criteria ?? {}
  const issues = response.groundedness_issues ?? []
  const hasCriteria = Object.keys(criteria).length > 0
  if (!isJudged) {
    return (
      <div className="pl-9">
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-500" title={meta.detail ?? undefined}>
          <ShieldCheck size={13} className={meta.iconClass} />
          <span className="font-medium text-slate-600">LLM Judge</span>
          <span className={clsx('font-semibold', meta.labelClass)}>{meta.label}</span>
        </div>
      </div>
    )
  }
  return (
    <div className="pl-9">
      <button
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors"
      >
        <ShieldCheck size={13} className={meta.iconClass} />
        <span className="font-medium text-slate-600">LLM Judge</span>
        <span className={clsx('font-semibold', meta.labelClass)}>{meta.label}</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div id={id} className="mt-2 bg-slate-50 border border-slate-200 rounded-xl px-3 py-2.5 space-y-2">
          {hasCriteria && (
            <div className="space-y-1.5">
              {Object.entries(criteria).map(([k, v]) => (
                <CriterionRow key={k} label={k} score={v} />
              ))}
            </div>
          )}
          {issues.length > 0 && (
            <div className={hasCriteria ? 'pt-2 border-t border-slate-200' : ''}>
              <p className="text-xs font-medium text-slate-500 mb-1">Unsupported claims</p>
              <ul className="space-y-1">
                {issues.map((issue, i) => (
                  <li key={i} className="text-xs text-red-600 flex gap-1.5">
                    <span className="flex-shrink-0 mt-0.5">•</span>
                    <span>{issue}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {issues.length === 0 && (
            <p className="text-xs text-green-700">No unsupported claims detected.</p>
          )}
        </div>
      )}
    </div>
  )
}


export default function Chat({
  datasetId,
  conversationId,
  onConversationChange,
  onResponse,
  onNewConversation,
}: ChatProps) {
  const qc = useQueryClient()
  const [message, setMessage] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [synthesizing, setSynthesizing] = useState(false)
  const [plannedTools, setPlannedTools] = useState<ToolCall[]>([])
  const [toolProgress, setToolProgress] = useState<ToolProgress[]>([])
  const [streamError, setStreamError] = useState<string | null>(null)
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const { data: history } = useQuery({
    queryKey: ['history', conversationId],
    queryFn: () => (conversationId ? getHistory(conversationId) : null),
    enabled: !!conversationId,
  })

  const turns = history?.turns ?? []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, toolProgress, streaming])

  useEffect(() => {
    if (!conversationId || streaming || lastResponse) return
    const latestAssistant = [...turns]
      .reverse()
      .find((turn) => turn.role === 'assistant' && (
        (turn.tool_results?.length ?? 0) > 0 ||
        (turn.tables?.length ?? 0) > 0 ||
        (turn.charts?.length ?? 0) > 0
      ))
    if (!latestAssistant) return
    onResponse({
      dataset_id: latestAssistant.dataset_id ?? null,
      conversation_id: conversationId,
      message: latestAssistant.content,
      tool_calls: latestAssistant.tool_calls as unknown as ToolCall[],
      tool_results: latestAssistant.tool_results ?? [],
      tables: latestAssistant.tables ?? [],
      charts: latestAssistant.charts ?? [],
      citations: latestAssistant.citations ?? [],
      llm_enabled: true,
      planning_source: latestAssistant.planning_source === 'llm' ? 'llm' : 'rules',
      synthesis_source: latestAssistant.synthesis_source === 'llm' ? 'llm' : 'rules',
      llm_notes: [],
      groundedness_score: latestAssistant.groundedness_score,
      groundedness_criteria: latestAssistant.groundedness_criteria ?? {},
      groundedness_issues: latestAssistant.groundedness_issues ?? [],
      judge_status: latestAssistant.judge_status ?? 'rule_based',
    })
  }, [conversationId, streaming, lastResponse, turns, onResponse])

  const handleStream = useCallback(async () => {
    if (!message.trim() || streaming) return
    setStreaming(true)
    setThinking(false)
    setSynthesizing(false)
    setStreamError(null)
    setPlannedTools([])
    setToolProgress([])
    setLastResponse(null)
    let receivedDone = false

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          message: message.trim(),
          top_k: 6,
          conversation_id: conversationId,
        }),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text)
      }

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          let event: SSEEvent
          try {
            event = JSON.parse(raw) as SSEEvent
          } catch {
            continue
          }

          if (event.type === 'thinking') {
            setThinking(true)
            setSynthesizing(false)
          } else if (event.type === 'synthesizing') {
            setSynthesizing(true)
            setThinking(false)
          } else if (event.type === 'plan') {
            setThinking(false)
            setSynthesizing(false)
            setPlannedTools(event.tool_calls)
            onConversationChange(event.conversation_id)
          } else if (event.type === 'tool_result') {
            setToolProgress((prev) => {
              const next = prev.filter((p) => p.name !== event.name)
              return [...next, { name: event.name, status: event.ok ? 'ok' : 'error', error: event.error }]
            })
          } else if (event.type === 'error') {
            setStreamError(event.detail)
          } else if (event.type === 'done') {
            receivedDone = true
            setLastResponse(event.response)
            onResponse(event.response)
            await qc.invalidateQueries({ queryKey: ['history', event.response.conversation_id] })
          }
        }
      }
    } catch (err) {
      // Browsers (especially Firefox) throw a network error when the server
      // closes the SSE connection after the final "done" event — even though
      // the stream completed successfully.  Suppress it in that case.
      if (!receivedDone) {
        setStreamError(err instanceof Error ? err.message : 'Request failed')
      }
    } finally {
      setStreaming(false)
      setThinking(false)
      setSynthesizing(false)
      setPlannedTools([])
      setToolProgress([])
      setMessage('')
    }
  }, [message, streaming, datasetId, conversationId, onConversationChange, onResponse, qc])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleStream()
    }
  }

  return (
    <div className="w-[420px] flex-shrink-0 flex flex-col h-full border-r border-slate-200 bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <MessageSquare size={16} className="text-slate-500" />
          <h2 className="text-slate-800 font-semibold text-sm">Chat</h2>
        </div>
        <button
          onClick={onNewConversation}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800 transition-colors"
          title="New conversation"
        >
          <Plus size={14} />
          New
        </button>
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto thin-scroll px-4 py-4 space-y-4">
        {turns.length === 0 && !streaming && (
          <p className="text-slate-400 text-sm text-center mt-8">
            No messages yet. Upload a dataset and start asking questions.
          </p>
        )}

        {turns.map((turn, i) => (
          <Message key={i} turn={turn} conversationId={conversationId} turnIdx={i} />
        ))}

        {streaming && thinking && !plannedTools.length && (
          <div className="flex gap-2.5">
            <div className="w-7 h-7 flex items-center justify-center">
              <Loader2 size={16} className="text-indigo-400 animate-spin" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm shadow-sm px-3.5 py-2.5 text-xs text-slate-400 italic">
              Planning…
            </div>
          </div>
        )}

        {streaming && !!plannedTools.length && (
          <ToolProgressList planned={plannedTools} progress={toolProgress} synthesizing={synthesizing} />
        )}

        {streamError && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-3.5 py-2.5 text-sm text-red-700">
            {streamError}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-slate-200 px-4 py-3 space-y-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
          disabled={streaming}
          placeholder={
            'Examples:\n• Analyse this dataset\n• Train a model to predict churn\n• sql: SELECT region, SUM(revenue) FROM t GROUP BY region'
          }
          className="w-full resize-none text-sm text-slate-800 placeholder-slate-400 border border-slate-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-60 disabled:bg-slate-50"
        />
        <div className="flex gap-2">
          <button
            onClick={handleStream}
            disabled={streaming || !message.trim()}
            className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {streaming ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Send size={14} />
            )}
            {streaming ? 'Running…' : 'Run'}
          </button>
          <p className="text-slate-400 text-xs self-center">Ctrl+Enter</p>
        </div>
      </div>
    </div>
  )
}
