r"""DYNAMIC BATCHING serving: requests that arrive together share ONE forward.

The batcher: each request drops its text into a queue and waits on a
future (an IOU for a result). A background worker takes what is in the
queue - waiting at most MAX_WAIT_MS for company, up to MAX_BATCH items -
runs ONE batched forward pass, then delivers each answer to its waiting
request.

Run from the project root:
    uvicorn scripts.serve_batched:app --port 8001
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.text_predictor import TextPredictor

MAX_BATCH = 32     # never run a batch bigger than this
MAX_WAIT_MS = 5    # never make a request wait longer than this for company


class DynamicBatcher:
    def __init__(self, predictor):
        self.predictor = predictor
        self.queue: asyncio.Queue = asyncio.Queue()

    async def predict(self, text: str) -> dict:
        """Called by endpoints: enqueue the text, wait for the answer."""
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def worker(self):
        """Runs forever: collect a batch, one forward pass, deliver answers."""
        loop = asyncio.get_running_loop()
        while True:
            # Block until at least one request exists...
            text, future = await self.queue.get()
            batch = [(text, future)]

            # ...then wait up to MAX_WAIT_MS for more to join.
            deadline = loop.time() + MAX_WAIT_MS / 1000
            while len(batch) < MAX_BATCH:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break

            texts = [item[0] for item in batch]
            # Run the heavy forward in the thread pool (Tutorial #2, Milestone 6:
            # never block the event loop with CPU work).
            results = await loop.run_in_executor(
                None, self.predictor.predict_batch, texts
            )
            for (_, fut), result in zip(batch, results):
                fut.set_result(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = TextPredictor("models/sentiment_transformer")
    app.state.batcher = DynamicBatcher(predictor)
    worker_task = asyncio.create_task(app.state.batcher.worker())
    print("model loaded (batched server)")
    yield
    worker_task.cancel()


app = FastAPI(title="Sentiment API - dynamic batching", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


@app.post("/predict")
async def predict(body: TextIn):
    return await app.state.batcher.predict(body.text)