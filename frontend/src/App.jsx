import React, { useState, useRef, useEffect } from 'react';
import { Send, GraduationCap, MapPin, Loader2, BookOpen, ExternalLink, Bot, User, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';

// --- Custom Markdown Components for beautiful formatting ---
const markdownComponents = {
  // Links: Blue and underlined
  a: ({node, ...props}) => (
    <a {...props} className="text-blue-400 hover:text-blue-300 underline font-medium transition-colors" target="_blank" rel="noopener noreferrer" />
  ),
  // Bold: Indigo color to make dates/deadlines pop
  strong: ({node, ...props}) => (
    <strong {...props} className="font-bold text-indigo-300" /> 
  ),
  // Lists: Proper spacing and marker color
  ul: ({node, ...props}) => (
    <ul {...props} className="list-disc ml-4 space-y-2 my-2 marker:text-indigo-400" />
  ),
  ol: ({node, ...props}) => (
    <ol {...props} className="list-decimal ml-4 space-y-2 my-2 marker:text-indigo-400" />
  ),
  li: ({node, ...props}) => (
    <li {...props} className="pl-1 leading-relaxed" />
  ),
  // Paragraphs: Ensure spacing
  p: ({node, ...props}) => (
    <p {...props} className="mb-2 last:mb-0 leading-relaxed" />
  ),
  // Blockquotes: Syling for notes/warnings
  blockquote: ({node, ...props}) => (
    <blockquote {...props} className="border-l-4 border-indigo-500 pl-4 py-1 my-3 bg-slate-800/30 italic text-slate-300 rounded-r" />
  )
};

// --- Typing Effect Component ---
const Typewriter = ({ text, onComplete }) => {
  const [displayLength, setDisplayLength] = useState(0);
  
  useEffect(() => {
    // Reset when text changes
    setDisplayLength(0);
    
    if (!text) return;

    let i = 0;
    const intervalId = setInterval(() => {
      i++;
      setDisplayLength((prev) => {
        if (prev >= text.length) {
          clearInterval(intervalId);
          if (onComplete) onComplete();
          return text.length;
        }
        return prev + 1;
      });
    }, 15); // Consistent typing speed

    return () => clearInterval(intervalId);
  }, [text]);

  // Derive the text to show based on current length
  const contentToShow = text.slice(0, displayLength);

  return (
    <ReactMarkdown 
      remarkPlugins={[remarkGfm]}
      className="prose prose-invert prose-sm max-w-none text-slate-100"
      components={markdownComponents}
    >
      {contentToShow}
    </ReactMarkdown>
  );
};

function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([
    { 
      id: 'welcome',
      role: 'assistant', 
      content: "Hallo! Ik ben de Howest International Bot. Ik kan al je vragen beantwoorden over buitenlandse stages, studies en procedures. Waarmee kan ik je helpen?", 
      sources: [] 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentTypingId, setCurrentTypingId] = useState(null); 
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // --- Session Management ---

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/sessions?user_id=student_default');
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        return data;
      }
    } catch (e) {
      console.error("Failed to load sessions", e);
    }
    return [];
  };

  const loadSession = async (sessionId) => {
    setIsLoading(true);
    setCurrentSessionId(sessionId);
    try {
      const res = await fetch(`http://localhost:8000/history?session_id=${sessionId}`);
      if (res.ok) {
        const history = await res.json();
        setMessages(history); 
      }
    } catch (e) {
      console.error("Failed to load history", e);
    } finally {
      setIsLoading(false);
    }
  };

  const createNewChat = () => {
    setCurrentSessionId(null);
    setMessages([{ 
      id: 'welcome',
      role: 'assistant', 
      content: "Nieuwe chat gestart! Waarmee kan ik je helpen?", 
      sources: [] 
    }]);
  };

  const deleteSession = async (e, sessionId) => {
    e.stopPropagation(); // Prevent triggering loadSession
    if (!window.confirm("Ben je zeker dat je deze chat wilt verwijderen?")) return;

    try {
      await fetch(`http://localhost:8000/sessions/${sessionId}`, { method: 'DELETE' });
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      
      // If we deleted the active session, go to new chat
      if (currentSessionId === sessionId) {
        createNewChat();
      }
    } catch (error) {
      console.error("Failed to delete session", error);
    }
  };

  // Initial Load
  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentTypingId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    let activeSessionId = currentSessionId;

    try {
      // 1. Create Session if needed
      if (!activeSessionId) {
        const sessionRes = await fetch('http://localhost:8000/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: 'student_default' })
        });
        const sessionData = await sessionRes.json();
        activeSessionId = sessionData.id;
        setCurrentSessionId(activeSessionId);
        // Refresh list to show new session
        fetchSessions(); 
      }

      // 2. Send Query
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: userMessage.content,
          session_id: activeSessionId // Pass session ID
        }),
      });

      if (!response.ok) throw new Error('Network error');
      
      const data = await response.json();
      
      // 3. Add assistant message
      const botMessageId = Date.now() + 1;
      setMessages(prev => [...prev, { 
        id: botMessageId, 
        role: 'assistant', 
        content: data.answer, 
        sources: data.sources || [],
        isTyping: true 
      }]);
      setCurrentTypingId(botMessageId);

      // 4. Refresh session list (titles might update)
      fetchSessions();

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        role: 'assistant', 
        content: "Sorry, er ging iets mis bij het ophalen van het antwoord.",
        sources: [] 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Callback when typing finishes
  const handleTypingComplete = (msgId) => {
    if (currentTypingId === msgId) {
      setCurrentTypingId(null);
      setMessages((prev) => 
        prev.map(m => m.id === msgId ? { ...m, isTyping: false } : m)
      );
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className="hidden md:flex w-72 flex-col bg-slate-900 border-r border-slate-800 p-4">
        <div className="flex items-center gap-3 mb-8 px-2">
          <div className="bg-red-600 p-2 rounded-lg text-white">
            <GraduationCap size={24} />
          </div>
          <div>
            <h1 className="font-bold text-lg">Go International</h1>
            <p className="text-xs text-slate-400">Student Assist</p>
          </div>
        </div>

        <button 
             onClick={createNewChat} 
             className="flex items-center gap-3 w-full px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-md transition-colors border border-dashed border-slate-700 mb-6"
        >
             <BookOpen size={18} />
             <span>+ Nieuwe Chat</span>
        </button>

        <div className="flex-1 overflow-y-auto space-y-2 pr-2 scrollbar-thin">
           <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 px-2">Recente Chats</h3>
           {sessions.map(session => (
             <div 
                key={session.id}
                onClick={() => loadSession(session.id)}
                className={`group flex items-center justify-between w-full px-3 py-2 text-sm rounded-md transition-colors cursor-pointer ${
                  currentSessionId === session.id ? 'bg-indigo-900/50 text-indigo-200' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
             >
                <div className="flex items-center gap-2 truncate">
                   <span className="truncate max-w-[160px]" title={session.title}>{session.title}</span>
                </div>
                <button 
                  onClick={(e) => deleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-900/50 hover:text-red-400 rounded transition-all"
                  title="Verwijder chat"
                >
                  <Trash2 size={14} />
                </button>
             </div>
           ))}
           
           {sessions.length === 0 && (
              <p className="text-xs text-slate-600 px-3 italic">Nog geen chats...</p>
           )}
        </div>

        <div className="mt-auto pt-4 border-t border-slate-800 text-xs text-slate-500">
          <p>&copy; 2024 Howest International</p>
          <p className="mt-1">Powered by Local RAG & LLama3</p>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative">
        
        {/* Header (Mobile) */}
        <header className="md:hidden flex items-center bg-slate-900 border-b border-slate-800 p-4">
           <GraduationCap className="text-red-500 mr-2" />
           <span className="font-bold">Howest Int.</span>
        </header>

        {/* Messages List */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scrollbar-thin">
          <div className="max-w-3xl mx-auto space-y-8 pb-4">
            {messages.map((msg) => (
              <motion.div 
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}
              >
                
                {/* Avatar (Bot) */}
                {msg.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center mt-1">
                    <Bot size={16} />
                  </div>
                )}

                <div className={`space-y-2 max-w-[85%] md:max-w-[75%]`}>
                  <div className={`p-4 rounded-2xl ${
                    msg.role === 'user' 
                      ? 'bg-slate-800 text-slate-100 rounded-br-none' 
                      : 'bg-[#151b2b] border border-slate-800 text-slate-100 rounded-bl-none shadow-sm'
                  }`}>
                    
                    {msg.role === 'user' ? (
                      <p>{msg.content}</p>
                    ) : (
                      <div className="min-h-[20px]">
                         {/* If typing, use Typewriter, otherwise show Markdown directly */}
                         {msg.isTyping ? (
                           <Typewriter text={msg.content} onComplete={() => handleTypingComplete(msg.id)} />
                         ) : (
                           <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              className="prose prose-invert prose-sm max-w-none text-slate-200"
                              components={markdownComponents}
                           >
                            {msg.content}
                           </ReactMarkdown>
                         )}
                      </div>
                    )}
                  </div>

                  {/* Sources Display */}
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                     <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                        className="flex flex-wrap gap-2 mt-2 ml-1"
                     >
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                          <BookOpen size={12} /> Bronnen:
                        </span>
                        {msg.sources.map((src, idx) => {
                          const fileName = src.split('/').pop() || src;
                          // src is relative path like 'modules/...'
                          // Backend mounts /data at /static
                          const fileUrl = `http://localhost:8000/static/${src}`; 
                          
                          return (
                            <a 
                              key={idx} 
                              href={fileUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700 hover:text-blue-300 transition-colors cursor-pointer" 
                              title={src}
                            >
                              <ExternalLink size={10} />
                              {fileName.length > 25 ? fileName.substring(0, 25) + '...' : fileName}
                            </a>
                          )
                        })}
                     </motion.div>
                  )}
                </div>

                {/* Avatar (User) */}
                {msg.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center mt-1">
                    <User size={16} />
                  </div>
                )}
              </motion.div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
               <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-4"
               >
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center">
                    <Bot size={16} />
                  </div>
                  <div className="flex items-center gap-2 p-4 rounded-2xl rounded-bl-none bg-[#151b2b] border border-slate-800">
                    <span className="text-sm text-slate-400 flex items-center gap-2">
                       <Loader2 className="animate-spin" size={16} />
                       Analyzing documents...
                    </span>
                  </div>
               </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-slate-900 border-t border-slate-800">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={handleSubmit} className="relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Stel je vraag over internationaal studeren..."
                disabled={isLoading}
                className="w-full bg-slate-800 text-white placeholder-slate-400 rounded-xl pl-4 pr-12 py-3.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 border border-slate-700 transition-all shadow-lg"
              />
              <button 
                type="submit"
                disabled={isLoading || !input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} />
              </button>
            </form>
            <p className="text-center text-slate-500 text-xs mt-3">
              AI kan fouten maken. Controleer altijd officiële bronnen via STUVO of de dienst Internationalisering.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
