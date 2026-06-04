# Updated LLM Rule RAG STP/NSTP Agent

This is a replacement version for the STP/NSTP agent.

Main changes:
- The STP/NSTP decision is done by the LLM using all 78 company rules embedded directly in the prompt.
- SQL retrieval does not retrieve rules. It retrieves only historical/past proposal data, medical data, questionnaire data, and underwriter remarks.
- Vector retrieval does not use `latest_vector_store_config.json`. It auto-detects the newest Chroma folder inside `vector_store/`.
- The GUI does not ask for data completeness score or document authenticity score because OCR + Verification Agent handles that before this agent receives JSON.
- Final output includes `loading_agent_input` for the next Loading Proposal Agent.

Copy these files into your project root:
- app.py
- src/config.py
- src/schemas.py
- src/utils.py
- src/company_rules_prompt.py
- src/sql_retriever.py
- src/vector_retriever.py
- src/output_builder.py
- src/stp_nstp_agent.py
- requirements.txt

Keep your existing:
- database/underwriting_system.db
- vector_store/chroma_underwriting_...
- .env with OPENROUTER_API_KEY
