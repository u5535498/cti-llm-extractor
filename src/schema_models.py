"""
Pydantic models for schema_v0.2.json validation.
"""

from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field, validator, AnyUrl
from pydantic import field_validator
import re

class IOC(BaseModel):
    value: str = Field(..., min_length=1)
    evidence_snippet: str = Field(..., min_length=1)
    source_context: Optional[Literal["observed", "detection_rule_only", "uncertain"]] = None

class DomainIOC(IOC):
    masked: bool = Field(default=False)
    phishing_template: bool = Field(default=False)

class HashIOC(IOC):
    hash_type: Literal["MD5", "SHA1", "SHA256", "SHA512", "SSDEEP"] = Field(...)

class Tool(BaseModel):
    name: str = Field(..., min_length=1)
    evidence_snippet: str = Field(..., min_length=1)

class DetectionSignature(BaseModel):
    yara_rules: List[Tool] = Field(default_factory=list)
    sigma_rules: List[Tool] = Field(default_factory=list)

class TTP(BaseModel):
    mitre_id: Optional[str] = Field(None, pattern=r"^T[0-9]{4}(\.[0-9]{1,3})?$")
    tactic: str = Field(..., min_length=1)
    technique: str = Field(..., min_length=1)
    evidence_snippet: str = Field(..., min_length=1)
    mapping_type: Optional[Literal["explicit", "inferred"]] = "explicit"
    mitre_confidence: Optional[Literal["high", "medium", "low"]] = None
    evidence_location: Optional[str] = None

class TimelineEvent(BaseModel):
    date: str = Field(..., pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    date_precision: Literal["day", "month", "quarter", "year", "approximate"] = "day"
    event: str = Field(..., min_length=1)
    evidence_snippet: str = Field(..., min_length=1)

class ThreatActor(BaseModel):
    name: str = Field(..., min_length=1)
    aliases: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(...)

class ReportMetadata(BaseModel):
    report_type: Literal["incident", "campaign", "strategic"] = Field(...)
    report_title: Optional[str] = None

class IndicatorsOfCompromise(BaseModel):
    ip_addresses: List[IOC] = Field(default_factory=list)
    domains: List[DomainIOC] = Field(default_factory=list)
    file_hashes: List[HashIOC] = Field(default_factory=list)
    urls: List[IOC] = Field(default_factory=list)
    http_paths: List[IOC] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    cves: List[IOC] = Field(default_factory=list)
    detection_signatures: DetectionSignature = Field(default_factory=DetectionSignature)

class CTIDocument(BaseModel):
    """Root schema_v0.2 model."""
    report_metadata: ReportMetadata
    threat_actor: Optional[ThreatActor] = None
    campaign_name: Optional[str] = None
    target_sectors: List[str] = Field(default_factory=list)
    indicators_of_compromise: IndicatorsOfCompromise
    ttps: List[TTP] = Field(default_factory=list)
    summary: str = Field(..., max_length=1500)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    
    @field_validator('target_sectors')
    @classmethod
    def validate_sectors(cls, v):
        if any(len(s.strip()) < 1 for s in v):
            raise ValueError("Sectors must be non-empty")
        return [s.strip() for s in v]
