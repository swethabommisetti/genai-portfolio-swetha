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
    date: Optional[str] = None  # e.g., "2023-10-05"
    time: Optional[str] = None  # e.g., "14:30" # 24-hour format
    payment_method: Optional[str] = None  # e.g., "Credit Card", "Cash"
    tax: Optional[float] = None  # total tax amount if available
