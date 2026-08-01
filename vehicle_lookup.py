#!/usr/bin/env python3
"""
Vehicle Intelligence Platform (CLI Edition)

Enterprise-grade utility for processing and inspecting vehicle registration 
data from upstream API providers. Built following standard Python best practices.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Table

# Initialize rich output terminal
console = Console()

# ==========================================
# CONFIGURATION MANAGEMENT
# ==========================================

class AppConfig:
    """Central application configuration settings."""
    VERSION: str = "3.0.0-ENTERPRISE"
    API_BASE_URL: str = "https://vehicleinfobyterabaap.vercel.app/lookup"
    TIMEOUT_SECONDS: float = 10.0
    USER_AGENT: str = "VehicleIntelligenceEngine/3.0.0 (Enterprise Linux x86_64)"
    
    BASE_DIR: Path = Path.cwd()
    RESULTS_DIR: Path = BASE_DIR / "results"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"


# ==========================================
# DOMAIN MODELS & VALIDATION
# ==========================================

class RCQueryRequest(BaseModel):
    """Input validation for Registration Certificate lookup query."""
    rc_number: str = Field(..., min_length=3, max_length=20, description="Cleaned Registration Number")

    @classmethod
    def sanitize(cls, raw_rc: str) -> RCQueryRequest:
        cleaned = "".join(e for e in raw_rc if e.isalnum()).upper()
        return cls(rc_number=cleaned)


class VehiclePayload(BaseModel):
    """Structured vehicle data payload schema."""
    raw_data: Dict[str, Any]
    response_time_ms: float
    is_cached: bool = False


# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================

class VehicleIntelligenceException(Exception):
    """Base exception for application domain errors."""


class NetworkAPIError(VehicleIntelligenceException):
    """Raised when upstream API requests fail."""


class StorageError(VehicleIntelligenceException):
    """Raised when filesystem reading or writing operations fail."""


# ==========================================
# LOGGING ENGINE
# ==========================================

class StructuredJSONFormatter(logging.Formatter):
    """Formats system logs as structured JSON for log-aggregator compatibility."""
    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_object)


def configure_logging(logs_directory: Path) -> logging.Logger:
    """Configures structured file logging."""
    logs_directory.mkdir(parents=True, exist_ok=True)
    log_file = logs_directory / "app.log.json"

    logger = logging.getLogger("vehicle_intelligence")
    logger.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(StructuredJSONFormatter())
    logger.addHandler(file_handler)

    return logger


# ==========================================
# DATA ACCESS & STORAGE LAYER
# ==========================================

class LocalStorageEngine:
    """Manages file persistence, cache lifecycle, and operational artifacts."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._initialize_environment()

    def _initialize_environment(self) -> None:
        """Ensures all required storage paths exist."""
        for path in (self.config.RESULTS_DIR, self.config.LOGS_DIR, self.config.CACHE_DIR):
            path.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def read_cache(self, rc_number: str) -> Optional[Dict[str, Any]]:
        """Reads and deserializes cached vehicle data."""
        cache_file = self.config.CACHE_DIR / f"{self._hash_key(rc_number)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read error for key {rc_number}: {e}")
        return None

    def write_cache(self, rc_number: str, data: Dict[str, Any]) -> None:
        """Persists payload into local cache directory."""
        cache_file = self.config.CACHE_DIR / f"{self._hash_key(rc_number)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Cache write error for key {rc_number}: {e}")

    def export_result(self, rc_number: str, data: Dict[str, Any]) -> Path:
        """Exports JSON result artifact to results repository."""
        target_file = self.config.RESULTS_DIR / f"{rc_number}.json"
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return target_file
        except Exception as e:
            self.logger.error(f"Export error for RC {rc_number}: {e}")
            raise StorageError(f"Failed to persist output record: {e}") from e


# ==========================================
# HTTP CLIENT API LAYER
# ==========================================

class VehicleAPIClient:
    """Asynchronous client responsible for HTTP querying against API endpoints."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    async def fetch_record(self, query: RCQueryRequest) -> VehiclePayload:
        """Executes an asynchronous HTTP GET request to upstream provider."""
        headers = {"User-Agent": self.config.USER_AGENT}
        params = {"rc": query.rc_number}

        async with httpx.AsyncClient(timeout=self.config.TIMEOUT_SECONDS) as client:
            start_time = datetime.now()
            try:
                response = await client.get(self.config.API_BASE_URL, params=params, headers=headers)
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                if response.status_code != 200:
                    raise NetworkAPIError(
                        f"Upstream service responded with Status Code {response.status_code}"
                    )

                data = response.json()
                return VehiclePayload(
                    raw_data=data,
                    response_time_ms=round(latency_ms, 2),
                    is_cached=False
                )

            except httpx.TimeoutException as err:
                self.logger.error(f"Timeout querying RC {query.rc_number}")
                raise NetworkAPIError("Upstream service query timed out.") from err
            except httpx.HTTPError as err:
                self.logger.error(f"HTTP Transport Error for RC {query.rc_number}: {err}")
                raise NetworkAPIError(f"Network transport failure: {err}") from err
            except json.JSONDecodeError as err:
                self.logger.error(f"Invalid JSON response for RC {query.rc_number}")
                raise NetworkAPIError("Malformed response payload received from endpoint.") from err


# ==========================================
# PRESENTATION & VIEW LAYER
# ==========================================

class CLIView:
    """Handles formatted terminal presentation components."""

    @staticmethod
    def render_header(version: str) -> None:
        console.rule("[bold cyan]Vehicle Intelligence Platform[/bold cyan]")
        console.print(
            f"[dim]Version {version} | Secure Administrative & Verification Engine[/dim]",
            justify="center"
        )
        console.rule()

    @staticmethod
    def render_disclaimer() -> None:
        notice = (
            "[bold red]LEGAL & ETHICAL NOTICE:[/bold red]\n"
            "This utility is authorized strictly for official compliance, auditing, and legal research.\n"
            "Unauthorized data acquisition or monitoring is strictly governed by privacy regulations."
        )
        console.print(Panel(notice, style="dim yellow", expand=False))

    @staticmethod
    def render_payload(rc_number: str, payload: VehiclePayload) -> None:
        """Renders payload attributes in an organized datagrid."""
        meta_info = f"Source: {'Local Cache' if payload.is_cached else 'Remote API'} | Latency: {payload.response_time_ms} ms"
        console.print(f"\n[bold dim]{meta_info}[/bold dim]\n")

        table = Table(
            title=f"Record Detail Matrix — [bold cyan]{rc_number}[/bold cyan]",
            show_header=True,
            header_style="bold cyan",
            expand=True
        )
        table.add_column("Property Key", style="bold white", width=26)
        table.add_column("Value / Attribute Details", style="green")

        for key, value in payload.raw_data.items():
            if key.startswith("_"):
                continue  # Skip private system metadata
            
            formatted_key = key.replace("_", " ").title()
            table.add_row(formatted_key, str(value) if value is not None else "N/A")

        console.print(table)


# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class ApplicationController:
    """Orchestrates lookup logic between storage, network, and UI modules."""

    def __init__(self) -> None:
        self.config = AppConfig()
        self.logger = configure_logging(self.config.LOGS_DIR)
        self.storage = LocalStorageEngine(self.config, self.logger)
        self.api_client = VehicleAPIClient(self.config, self.logger)

    async def run_lookup(self, raw_rc: str) -> None:
        """Executes search workflow."""
        try:
            query = RCQueryRequest.sanitize(raw_rc)
        except ValidationError:
            console.print("[bold red]Validation Error:[/bold red] Invalid registration number format.")
            sys.exit(1)

        self.logger.info(f"Initiated transaction query for: {query.rc_number}")

        # 1. Attempt Cache Resolution
        cached_data = self.storage.read_cache(query.rc_number)
        if cached_data:
            self.logger.info(f"Cache hit for RC: {query.rc_number}")
            payload = VehiclePayload(raw_data=cached_data, response_time_ms=0.0, is_cached=True)
            CLIView.render_payload(query.rc_number, payload)
            return

        # 2. Remote Fetching
        try:
            with console.status("[bold cyan]Querying enterprise registry...[/bold cyan]", spinner="dots"):
                payload = await self.api_client.fetch_record(query)

            # Persist artifacts
            self.storage.write_cache(query.rc_number, payload.raw_data)
            saved_path = self.storage.export_result(query.rc_number, payload.raw_data)

            self.logger.info(f"Successfully processed query for {query.rc_number}. File: {saved_path}")
            
            CLIView.render_payload(query.rc_number, payload)
            console.print(f"[dim]Artifact saved to path: {saved_path}[/dim]\n")

        except VehicleIntelligenceException as err:
            console.print(Panel(f"[bold red]Processing Error:[/bold red] {err}", title="System Exception", style="red"))
            self.logger.error(f"Operational failure processing {query.rc_number}: {err}")
            sys.exit(1)


# ==========================================
# ENTRY POINT
# ==========================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle Intelligence Engine CLI")
    parser.add_argument(
        "--rc",
        type=str,
        help="Target Vehicle Registration Code (e.g., DL01AB1234)",
        required=False
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_cli_args()
    controller = ApplicationController()

    CLIView.render_header(AppConfig.VERSION)
    CLIView.render_disclaimer()

    rc_input = args.rc
    if not rc_input:
        try:
            rc_input = console.input("\n[bold cyan]Input Registration Code:[bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Execution aborted by user session.[/yellow]")
            sys.exit(0)

    await controller.run_lookup(rc_input)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold red]Terminated by user signal.[/bold red]")
        sys.exit(130)


if __name__ == "__main__":
    main()
