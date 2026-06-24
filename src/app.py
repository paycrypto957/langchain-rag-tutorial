import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag import invoke_rag, chat_history

# 创建 FastAPI 实例
app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="templates"))


@app.get("/")
async def homepage():
    return FileResponse("templates/naive_index.html")


@app.get("/stream_response")
async def stream_response(query: str):
    return StreamingResponse(
        invoke_rag(query, 1, chat_history), media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)
