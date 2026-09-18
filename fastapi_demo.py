"""FastAPI demo API layer backed by product logic."""

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from product import Product, ProductOut
import product as product_logic


app = FastAPI(title="FastAPI Demo", version="0.2.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello FastAPI"}


@app.get("/products", response_model=list[ProductOut])
def list_products() -> list[ProductOut]:
    return product_logic.list_products()


@app.get("/products/batch", response_model=list[Optional[ProductOut]])
async def get_products_batch(product_ids: str) -> list[Optional[ProductOut]]:
    ids = []
    for item in product_ids.split(","):
        if item.strip():
            ids.append(int(item))
    return await product_logic.get_products_by_ids(ids)


@app.get("/products/{product_id}", response_model=ProductOut)
async def get_product(product_id: int) -> ProductOut:
    found = await product_logic.get_product(product_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return found


@app.post("/products/add", response_model=ProductOut, status_code=201)
def add_product(payload: Product) -> ProductOut:
    return product_logic.add_product(payload)


if __name__ == "__main__":
    uvicorn.run("fastapi_demo:app", host="127.0.0.1", port=8000, reload=True)
