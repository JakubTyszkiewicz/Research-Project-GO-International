import React, { useState, useRef, useEffect } from 'react';
import { Send, GraduationCap, MapPin, Loader2, BookOpen, ExternalLink, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';

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
      components={{
        a: ({node, ...props}) => <a {...props} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer" />
      }}
    >
      {contentToShow}
    </ReactMarkdown>
  );
};

function App() {
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
  const [currentTypingId, setCurrentTypingId] = useState(null); // ID of message currently typing
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

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

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content }),
      });

      if (!response.ok) throw new Error('Network error');
      
      const data = await response.json();
      
      // Add assistant message but mark it as "typing"
      const botMessageId = Date.now() + 1;
      setMessages(prev => [...prev, { 
        id: botMessageId, 
        role: 'assistant', 
        content: data.answer, 
        sources: data.sources || [],
        isTyping: true 
      }]);
      setCurrentTypingId(botMessageId);

    } catch (error) {
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        role: 'assistant', 
        content: "Sorry, er ging iets mis bij het ophalen van het antwoord. Controleer of de backend draait.",
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

        <nav className="flex-1 space-y-2">
          <button className="flex items-center gap-3 w-full px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-md transition-colors">
            <MapPin size={18} />
            <span>Bestemmingen</span>
          </button>
          <button className="flex items-center gap-3 w-full px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-md transition-colors">
            <BookOpen size={18} />
            <span>Procedures</span>
          </button>
        </nav>

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
                              components={{
                                a: ({node, ...props}) => <a {...props} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer" />
                              }}
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
