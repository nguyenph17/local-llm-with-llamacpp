import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8091,
        proxy_headers=True,
        reload=True,
        workers=10,
    )