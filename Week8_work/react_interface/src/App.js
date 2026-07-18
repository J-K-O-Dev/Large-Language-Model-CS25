import React, { useState, useEffect, useRef } from "react";
import "./App.css";
const initialMessage = {
  text: "Hello! I'm a chatbot. How can I help you today?",
  sender: "bot",
};

function App() {
  // Lazily initialize state from localStorage
  const [messages, setMessages] = useState(() => {
    try {
      const savedMessages = localStorage.getItem("chatHistory");
      return savedMessages ? JSON.parse(savedMessages) : [initialMessage];
    } catch (error) {
      console.error("Failed to parse chat history from localStorage", error);
      return [initialMessage];
    }
  });
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messageListRef = useRef(null);

  // Effect to scroll to the bottom of the message list when new messages are added
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Effect to save chat history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() && !isLoading) {
      const userMessage = { text: input, sender: "user" };
      const newMessages = [...messages, userMessage];

      // Update UI immediately with user's message
      setMessages(newMessages);
      setInput("");
      setIsLoading(true);

      // --- API Call to your Python/Ollama Backend ---
      // This is where you make a POST request to your backend server.
      // The backend will run your RAG pipeline from the notebook.
      try {
        const response = await fetch("http://localhost:5000/chat", {
          // Assuming your backend is at this address
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            // 'Authorization': `Bearer YOUR_API_KEY` // Add API key here if needed
          },
          body: JSON.stringify({ history: newMessages }), // Send the whole conversation history for context
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const botMessage = { text: data.reply, sender: "bot" };

        setMessages((prevMessages) => [...prevMessages, botMessage]);
      } catch (error) {
        console.error("Error fetching bot response:", error);
        const errorMessage = {
          text: "Sorry, I'm having trouble connecting to the bot. Please make sure the backend server is running.",
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleClearChat = () => {
    setMessages([initialMessage]);
    // The useEffect hook will automatically update localStorage.
  };

  return (
    <div className="App">
      <div className="chat-container">
        <header className="chat-header">
          <h1>LLM Chat Interface</h1>
          <button onClick={handleClearChat} className="clear-button">
            Clear Chat
          </button>
        </header>
        <div className="message-list" ref={messageListRef}>
          {messages.map((message, index) => (
            <div key={index} className={`message-bubble ${message.sender}`}>
              <p>{message.text}</p>
            </div>
          ))}
          {isLoading && (
            <div className="message-bubble bot">
              <div className="loading-dots">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
        </div>
        <div className="message-input-container">
          <input
            type="text"
            className="message-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask a question..."
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            className="send-button"
            disabled={isLoading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
