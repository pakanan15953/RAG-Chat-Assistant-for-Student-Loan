from langfuse import get_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

# สร้าง client
lf = get_client()
lf_handler = LangfuseCallbackHandler()

trace_id = "unique-id-1234"  # หรือ uuid.uuid4()
user_prompt = "Brandon is 33"

with lf.start_as_current_span(name="ollama_video_trace") as span:
    # บันทึก input
    span.update_trace(input={"user_prompt": user_prompt})

    # เรียก Ollama model
    import ollama
    instruction_prompt = f"Extract name and age from the following text:\n{user_prompt}"
    output = ollama.generate("llama3.2:latest", prompt=instruction_prompt)

    # บันทึก output
    span.update_trace(output={"response": output.get("response")})

    # ส่งข้อมูลไป Langfuse
    lf.flush()

print(output.get("response"))