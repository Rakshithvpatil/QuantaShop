"""
ingestion/erp_service.py
────────────────────────
Mock ERP (Enterprise Resource Planning) service.
Mimics SAP / NetSuite inventory + purchase-order endpoints
using a local SQLite database.

Run:  uvicorn ingestion.erp_service:app --port 8001 --reload
Docs: http://localhost:8001/docs
"""

import sqlite3
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── Database path ─────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "erp.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory for dict-style access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def seed_database() -> None:
    """Create tables and insert sample SKU inventory on first run."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inventory (
            sku         TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            qty         INTEGER DEFAULT 0,
            reorder_pt  INTEGER DEFAULT 50,
            warehouse   TEXT DEFAULT 'TX-01',
            unit_cost   REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS purchase_orders (
            po_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            sku         TEXT NOT NULL,
            qty_ordered INTEGER NOT NULL,
            supplier    TEXT,
            status      TEXT DEFAULT 'pending',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        INSERT OR IGNORE INTO inventory VALUES
          ('SKU-001', 'Smart Glasses Gen1',  120, 30, 'TX-01', 299.99),
          ('SKU-002', 'Charging Case',       450, 100,'TX-01',  49.99),
          ('SKU-003', 'Replacement Lens',    230, 75, 'CA-02',  29.99),
          ('SKU-004', 'Firmware USB Dongle', 80,  20, 'TX-01',  19.99),
          ('SKU-005', 'Sensor Module v2',    15,  25, 'TX-01', 149.99);
    """)
    conn.commit()
    conn.close()


# ── Seed on import ────────────────────────────────────────────
seed_database()

app = FastAPI(title="SwiftPulse Mock ERP", version="1.0.0")


# ── Schemas ───────────────────────────────────────────────────
class InventoryUpdate(BaseModel):
    qty: int
    warehouse: Optional[str] = None


class POCreate(BaseModel):
    sku: str
    qty_ordered: int
    supplier: Optional[str] = "AutoSupplier"


# ── Routes ────────────────────────────────────────────────────
@app.get("/inventory")
def list_inventory():
    """Return all SKU stock levels."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM inventory").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/inventory/{sku}")
def get_inventory(sku: str):
    """Return stock level for a single SKU."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM inventory WHERE sku = ?", (sku,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"SKU {sku} not found")
    return dict(row)


@app.patch("/inventory/{sku}")
def update_inventory(sku: str, update: InventoryUpdate):
    """Update quantity (called after a BigCommerce order ships)."""
    conn = get_conn()
    conn.execute(
        "UPDATE inventory SET qty = ?, warehouse = COALESCE(?, warehouse) WHERE sku = ?",
        (update.qty, update.warehouse, sku)
    )
    conn.commit()
    conn.close()
    return {"status": "updated", "sku": sku}


@app.post("/purchase_orders")
def create_po(po: POCreate):
    """Auto-create a purchase order when stock hits reorder point."""
    conn = get_conn()
    cursor = conn.execute(
        "INSERT INTO purchase_orders (sku, qty_ordered, supplier) VALUES (?,?,?)",
        (po.sku, po.qty_ordered, po.supplier)
    )
    conn.commit()
    po_id = cursor.lastrowid
    conn.close()
    return {"status": "created", "po_id": po_id}


@app.get("/purchase_orders")
def list_pos():
    """Return all purchase orders."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM purchase_orders").fetchall()
    conn.close()
    return [dict(r) for r in rows]
