from datetime import datetime
from typing import Dict
from .models import Order, Side, OrderType

class FIXEngine:
    """
    Standard FIX 4.4 Protocol Engine for Production Integration.
    Maps simulator orders/trades to FIX tag-value messages.
    """
    def __init__(self, sender_comp_id: str = "SIM_AGENT", target_comp_id: str = "EXCHANGE"):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.msg_seq_num = 1

    def _get_header(self, msg_type: str) -> str:
        header = {
            8: "FIX.4.4",
            35: msg_type,
            49: self.sender_comp_id,
            56: self.target_comp_id,
            34: self.msg_seq_num,
            52: datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        }
        self.msg_seq_num += 1
        return "|".join([f"{k}={v}" for k, v in header.items()])

    def create_new_order(self, order: Order) -> str:
        """FIX MsgType D: NewOrderSingle"""
        body = {
            11: order.order_id,
            55: order.asset_id,
            54: 1 if order.side == Side.BUY else 2,
            40: 2 if order.order_type == OrderType.LIMIT else 1,
            38: order.quantity,
            44: order.price if order.order_type == OrderType.LIMIT else 0,
            60: datetime.fromtimestamp(order.timestamp).strftime("%Y%m%d-%H:%M:%S")
        }
        header = self._get_header("D")
        body_str = "|".join([f"{k}={v}" for k, v in body.items()])
        return f"{header}|{body_str}|10=000|"

    def create_execution_report(self, order: Order, last_qty: int, last_px: float) -> str:
        """FIX MsgType 8: ExecutionReport"""
        body = {
            37: f"EXEC_{order.order_id}",
            11: order.order_id,
            17: f"TRD_{datetime.now().timestamp()}",
            150: 2, # Fill
            39: 2 if order.remaining_quantity == 0 else 1, # Status: Filled or Partially Filled
            55: order.asset_id,
            54: 1 if order.side == Side.BUY else 2,
            38: order.quantity,
            32: last_qty,
            31: last_px,
            151: order.remaining_quantity,
            6: (order.quantity - order.remaining_quantity) * last_px / order.quantity # AvgPx proxy
        }
        header = self._get_header("8")
        body_str = "|".join([f"{k}={v}" for k, v in body.items()])
        return f"{header}|{body_str}|10=000|"
