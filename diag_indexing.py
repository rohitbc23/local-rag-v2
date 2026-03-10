from rag_engine import LocalRAG
import time

rag = LocalRAG()
stats = rag.get_indexed_stats()
print(f"Current chunks: {stats['total_chunks']}")

print("Waiting 5 seconds to check for growth...")
time.sleep(5)

stats2 = rag.get_indexed_stats()
print(f"New chunks: {stats2['total_chunks']}")

if stats2['total_chunks'] > stats['total_chunks']:
    print(f"GROWING! Added {stats2['total_chunks'] - stats['total_chunks']} chunks.")
else:
    print("STUCK! Chunk count is the same.")
