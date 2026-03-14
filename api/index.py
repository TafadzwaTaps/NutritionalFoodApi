from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Nutrition API running"}

@app.get("/health")
def health():
    return {"status": "ok"}
