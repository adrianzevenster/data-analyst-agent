import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { MessageSquare, Send, Plus, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'
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

function Message({ turn }: { turn: ConversationTurn }) {
  const isUser = turn.role === 'user'
  return (
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
        {turn.content}
      </div>
    </div>
  )
}

function ToolProgressList({
  planned,
  progress,
}: {
  planned: ToolCall[]
  progress: ToolProgress[]
}) {
  if (!planned.length) return null

  const progressMap = new Map(progress.map((p) => [p.name, p]))

  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 flex items-center justify-center">
        <Loader2 size={16} className="text-indigo-400 animate-spin" />
      </div>
      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm shadow-sm px-3.5 py-2.5 space-y-1">
        {planned.map((tc) => {
          const p = progressMap.get(tc.name)
          return (
            <div key={tc.name} className="flex items-center gap-2 text-xs">
              {!p ? (
                <div className="w-3 h-3 rounded-full border-2 border-slate-300 flex-shrink-0" />
              ) : p.status === 'ok' ? (
                <CheckCircle2 size={13} className="text-green-500 flex-shrink-0" />
              ) : (
                <XCircle size={13} className="text-red-500 flex-shrink-0" />
              )}
              <span className={clsx('font-mono', !p ? 'text-slate-400' : p.status === 'ok' ? 'text-slate-700' : 'text-red-600')}>
                {tc.name}
              </span>
              {p?.error && <span className="text-red-500 truncate max-w-[160px]">{p.error}</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function SynthesisBadge({ response }: { response: ChatResponse }) {
  if (!response.llm_enabled) return null
  return (
    <div className="flex justify-start pl-9">
      <span
        className={clsx(
          'text-xs px-2 py-0.5 rounded-full font-medium',
          response.synthesis_source === 'llm'
            ? 'bg-indigo-100 text-indigo-700'
            : 'bg-slate-100 text-slate-500'
        )}
      >
        {response.synthesis_source === 'llm' ? 'LLM synthesis' : 'Rule-based'}
      </span>
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

  const handleStream = useCallback(async () => {
    if (!message.trim() || streaming) return
    setStreaming(true)
    setStreamError(null)
    setPlannedTools([])
    setToolProgress([])
    setLastResponse(null)

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

          if (event.type === 'plan') {
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
            setLastResponse(event.response)
            onResponse(event.response)
            await qc.invalidateQueries({ queryKey: ['history', event.response.conversation_id] })
          }
        }
      }
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setStreaming(false)
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
          <Message key={i} turn={turn} />
        ))}

        {streaming && (
          <ToolProgressList planned={plannedTools} progress={toolProgress} />
        )}

        {streamError && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-3.5 py-2.5 text-sm text-red-700">
            {streamError}
          </div>
        )}

        {lastResponse && !streaming && (
          <SynthesisBadge response={lastResponse} />
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
