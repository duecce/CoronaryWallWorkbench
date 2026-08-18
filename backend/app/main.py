from fastapi import FastAPI

app = FastAPI(
    title="CoronaryWallWorkbench API",
    version="0.1.0",
    description="API for coronary case loading, geometry processing, and assisted wall annotation.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
