"""Product business logic with an in-memory store."""

import asyncio
from typing import Optional

from pydantic import BaseModel, Field


class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    price: float = Field(..., gt=0)
    in_stock: bool = True


class ProductOut(Product):
    id: int


_store: dict[int, Product] = {}
_next_id = 1


def _to_out(product_id: int) -> ProductOut:
    product = _store[product_id]
    return ProductOut(
        id=product_id,
        **product.model_dump(),
    )


def add_product(product: Product) -> ProductOut:
    global _next_id
    product_id = _next_id
    _store[product_id] = product
    _next_id += 1
    return _to_out(product_id)


def list_products() -> list[ProductOut]:
    result = []
    for product_id in sorted(_store):
        result.append(_to_out(product_id))
    return result


async def get_product(product_id: int) -> Optional[ProductOut]:
    if product_id not in _store:
        return None
    return _to_out(product_id)


async def get_products_by_ids(product_ids: list[int]) -> list[Optional[ProductOut]]:
    tasks = []
    for product_id in product_ids:
        task = asyncio.create_task(get_product(product_id))
        tasks.append(task)
    return await asyncio.gather(*tasks)
