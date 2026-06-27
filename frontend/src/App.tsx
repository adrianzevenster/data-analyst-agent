import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import Results from './components/Results'
import type { ChatResponse } from './types/api'

export default function App() {
  const [datasetId, setDatasetId] = useState<string | null>(null)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null)

  function handleNewConversation() {
    setConversationId(null)
    setLastResponse(null)
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        datasetId={datasetId}
        onDatasetChange={setDatasetId}
        conversationId={conversationId}
      />
      <div className="flex flex-1 overflow-hidden min-w-0">
        <Chat
          datasetId={datasetId}
          conversationId={conversationId}
          onConversationChange={setConversationId}
          onResponse={setLastResponse}
          onNewConversation={handleNewConversation}
        />
        <Results response={lastResponse} conversationId={conversationId} datasetId={datasetId} />
      </div>
    </div>
  )
}
