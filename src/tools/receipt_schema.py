from typing import List, Optional
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    quantity: Optional[float] = None  # in case weight or count is given
    cost: float

class ReceiptData(BaseModel):
    store_name: str
    address: str
    items: List[Item]
    total_cost: float
    number_of_items: int
