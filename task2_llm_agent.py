"""
Task 2: Simple LLM Agent (Debloated + Enhanced Memory)

Clean agent with real Groq LLM and smart memory management.
"""

from groq import Groq
import json
import time

# Groq API key
GROQ_API_KEY = "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"

class LLMAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.memory = []  # Enhanced memory storage
        self.context_window = 5  # Remember last 5 interactions
        
    def is_creative(self, text):
        """Detect creative requests"""
        creative_words = ['write', 'create', 'generate', 'story', 'poem', 'caption', 'imagine', 'make']
        return any(word in text.lower() for word in creative_words)
    
    def add_to_memory(self, user_input, response, intent):
        """Enhanced memory with metadata"""
        memory_entry = {
            'timestamp': time.time(),
            'user': user_input,
            'agent': response,
            'intent': intent,
            'length': len(response)
        }
        
        self.memory.append(memory_entry)
        
        # Keep only recent interactions
        if len(self.memory) > self.context_window:
            self.memory.pop(0)
    
    def build_context(self):
        """Build smart context from memory"""
        if not self.memory:
            return ""
        
        # Get recent context, prioritizing relevant interactions
        context_parts = []
        for entry in self.memory[-3:]:  # Last 3 interactions
            context_parts.append(f"User: {entry['user'][:50]}...")
            context_parts.append(f"Agent: {entry['agent'][:50]}...")
        
        return " | ".join(context_parts)
    
    def process_input(self, user_input):
        """Main processing with enhanced memory"""
        intent = "creative" if self.is_creative(user_input) else "factual"
        context = self.build_context()
        
        # Build prompt with memory context
        if intent == "creative":
            prompt = f"""You are a creative AI assistant. Be imaginative and engaging.

Previous conversation context: {context}

Creative request: {user_input}

Generate a creative response:"""
            temperature = 0.8
        else:
            prompt = f"""You are a helpful AI assistant. Give direct, accurate answers.

Previous conversation context: {context}

Question: {user_input}

Answer:"""
            temperature = 0.2
        
        # Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=temperature
            )
            
            agent_response = response.choices[0].message.content.strip()
            
            # Add to enhanced memory
            self.add_to_memory(user_input, agent_response, intent)
            
            return agent_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_to_memory(user_input, error_msg, "error")
            return error_msg
    
    def get_memory_summary(self):
        """Get detailed memory summary"""
        if not self.memory:
            return {"total": 0, "recent": []}
        
        factual_count = sum(1 for m in self.memory if m['intent'] == 'factual')
        creative_count = sum(1 for m in self.memory if m['intent'] == 'creative')
        
        recent_topics = [m['user'][:30] + "..." if len(m['user']) > 30 
                        else m['user'] for m in self.memory[-3:]]
        
        return {
            "total": len(self.memory),
            "factual": factual_count,
            "creative": creative_count,
            "recent_topics": recent_topics,
            "memory_entries": self.memory
        }
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory = []
    
    def export_memory(self):
        """Export memory as JSON"""
        return json.dumps(self.memory, indent=2)

def interactive_chat():
    """Enhanced interactive chat"""
    print("🤖 LLM Agent - Enhanced Memory Chat")
    print("=" * 45)
    print("Commands: quit, clear, summary, memory, export")
    print("=" * 45)
    
    agent = LLMAgent()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                print("🧹 Memory cleared!")
                continue
                
            elif user_input.lower() == 'summary':
                summary = agent.get_memory_summary()
                print(f"\n📊 Memory Summary:")
                print(f"   Total: {summary['total']} interactions")
                print(f"   Factual: {summary['factual']}, Creative: {summary['creative']}")
                print(f"   Recent: {summary['recent_topics']}")
                continue
                
            elif user_input.lower() == 'memory':
                summary = agent.get_memory_summary()
                print(f"\n🧠 Detailed Memory:")
                for i, entry in enumerate(summary['memory_entries'], 1):
                    print(f"   {i}. [{entry['intent']}] {entry['user'][:40]}...")
                continue
                
            elif user_input.lower() == 'export':
                memory_json = agent.export_memory()
                print(f"\n💾 Memory Export:\n{memory_json}")
                continue
                
            elif not user_input:
                continue
            
            # Process input
            print("🤔 Thinking...")
            response = agent.process_input(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def demo_agent():
    """Quick demo"""
    print("=== LLM Agent Demo ===")
    
    agent = LLMAgent()
    
    queries = [
        "Who is the CEO of Google?",
        "Write a haiku about coding",
        "What did we just talk about?",  # Tests memory
        "Create a story based on our conversation"  # Tests memory
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = agent.process_input(query)
        print(f"Agent: {response}")
    
    # Show memory
    summary = agent.get_memory_summary()
    print(f"\nMemory: {summary['total']} interactions stored")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':
        interactive_chat()
    else:
        demo_agent()