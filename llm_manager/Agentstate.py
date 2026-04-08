"""
SciAgent State Management System

Structured state (not conversational memory) for a precise tool-calling AI agent.
The agent is always "informed" (knows purposes, file locations, routing decisions).
Eliminates repetition and enables context-aware decision making.
"""

from typing import TypedDict, Optional, Dict, List, Any, Literal
from pathlib import Path

# ============================================================================
# Core Agent State (The "Clipboard")
# ============================================================================

class AgentState(TypedDict):
    """
    Persistent state for SciAgent across tool calls.
    
    This replaces conversational memory with structured tracking:
    - File locations (no repeated "where is the data?" questions)
    - Triage results (knows if standard vs map workflow)
    - Quality flags (aware of data issues before processing)
    - Pipeline progress (knows what's done, what's next)
    """
    
    # ========== File System Context ==========
    current_data_path: Optional[str]           # Active data folder
    current_template_path: Optional[str]       # Peak fitting templates
    output_directory: Optional[str]            # Where results go
    current_file: Optional[str]                # Single file being processed
    
    # ========== Data Triage & Routing ==========
    # Populated by triage_decision_node
    data_type: Optional[str]                   # "standard_spectra" | "map_hyperspectral" | "map_2d"
    recommended_workflow: Optional[str]        # "standard" | "map"
    triage_confidence: Optional[float]         # 0.0-1.0 confidence in detection
    triage_reason: Optional[str]               # Why this workflow was chosen
    triage_parameters: Optional[Dict[str, Any]]  # Detected: nx, ny, region, energy_points
    
    # ========== Quality Metadata ==========
    # Gatekeeper: flags issues BEFORE heavy processing
    quality_flags: Optional[Dict[str, Any]]    # SNR, peak quality, energy shifts
    low_quality_samples: Optional[List[str]]   # Files flagged as problematic
    user_quality_decision: Optional[str]       # "skip" | "process_anyway" | "manual_review"
    
    # ========== Pipeline State ==========
    # Tracks what's done to avoid repetition
    pipeline_stage: Optional[str]              # Current stage in workflow
    completed_steps: List[str]                 # ["grouping", "reader", "fitter", ...]
    pending_steps: List[str]                   # ["quantifier", "plotter", ...]
    last_analysis_summary: Optional[str]       # Quick summary of results
    
    # ========== Grouping Context ==========
    current_groups: Optional[Dict[str, List[str]]]  # {"A": [files], "B": [files]}
    group_strategy: Optional[str]              # "by_filename" | "user_defined" | "auto"
    
    # ========== Drift & Anomaly Detection ==========
    # Vigilant Observer: tracks experimental issues
    detected_drift: Optional[Dict[str, Any]]   # Energy calibration shifts
    outlier_samples: Optional[List[str]]       # Statistical anomalies flagged
    
    # ========== Chat History (for context if needed) ==========
    messages: List[Dict[str, str]]             # [{"role": "user", "content": "..."}, ...]
    
    # ========== Tool Call Tracking ==========
    last_tool_call: Optional[str]              # Name of last tool executed
    last_tool_result: Optional[Dict[str, Any]] # Full result from last tool
    user_alerts: List[str]                     # Accumulated warnings/issues to tell user


# ============================================================================
# State Update Functions
# ============================================================================

def update_data_path(state: AgentState, path: str) -> AgentState:
    """
    Update current data path when user provides new location.
    
    Example:
        User: "Use the data in project_root/00_raw_data"
        Agent: Calls this function to remember the path
    """
    state['current_data_path'] = str(Path(path).resolve())
    return state


def update_triage_results(state: AgentState, triage_result: Dict[str, Any]) -> AgentState:
    """
    Update state with triage decision results.
    
    Called automatically after triage_decision_node runs.
    The agent becomes aware of:
    - Data type (standard vs map)
    - Recommended workflow
    - Detected parameters (dimensions, regions, etc.)
    """
    state['data_type'] = triage_result['data_type'].value if hasattr(triage_result['data_type'], 'value') else triage_result['data_type']
    state['triage_confidence'] = triage_result['confidence']
    state['triage_reason'] = triage_result['reason']
    state['triage_parameters'] = triage_result.get('parameters', {})
    state['recommended_workflow'] = triage_result.get('recommended_processor', 'unknown')
    return state


def add_quality_flag(state: AgentState, sample: str, issue: str, severity: str = "warning") -> AgentState:
    """
    Flag quality issues before processing (Gatekeeper function).
    
    Example:
        add_quality_flag(state, "sample_46.vms", "SNR < 3 (very low)", "error")
        
    Agent will alert user: "⚠️ Sample 46: SNR is very low (<3). Skip or process anyway?"
    """
    if state['quality_flags'] is None:
        state['quality_flags'] = {}
    
    if state['low_quality_samples'] is None:
        state['low_quality_samples'] = []
    
    state['quality_flags'][sample] = {"issue": issue, "severity": severity}
    
    if severity in ["error", "critical"]:
        state['low_quality_samples'].append(sample)
    
    return state


def add_user_alert(state: AgentState, message: str) -> AgentState:
    """
    Add alert that must be shown to user.
    
    These accumulate and are displayed in the next agent response.
    Used for tool warnings, quality issues, drift detection, etc.
    """
    if state['user_alerts'] is None:
        state['user_alerts'] = []
    
    state['user_alerts'].append(message)
    return state


def mark_step_complete(state: AgentState, step_name: str, summary: str = "") -> AgentState:
    """
    Mark a pipeline step as completed.
    
    Prevents redundant work and tracks progress.
    
    Example:
        mark_step_complete(state, "xps_reader", "Read 50 spectra, all valid")
    """
    if state['completed_steps'] is None:
        state['completed_steps'] = []
    
    if step_name not in state['completed_steps']:
        state['completed_steps'].append(step_name)
    
    # Remove from pending if it was there
    if state['pending_steps'] and step_name in state['pending_steps']:
        state['pending_steps'].remove(step_name)
    
    if summary:
        state['last_analysis_summary'] = summary
    
    return state


def detect_drift(state: AgentState, drift_data: Dict[str, Any]) -> AgentState:
    """
    Record detected energy drift (Vigilant Observer function).
    
    Example drift_data:
        {
            "samples": ["sample_1", ..., "sample_58"],
            "stable_range": [0, 58],
            "drift_start": 59,
            "reference_peak": 284.8,
            "shifted_to": 290.4,
            "severity": "critical"
        }
    
    Agent will alert: "⚠️ Energy drift detected starting at Sample 59!"
    """
    state['detected_drift'] = drift_data
    
    alert = (
        f"⚠️ Energy Drift Detected!\n"
        f"  Stable samples: {drift_data['stable_range'][0]}-{drift_data['stable_range'][1]} "
        f"(reference C1s at {drift_data['reference_peak']} eV)\n"
        f"  Drift starts: Sample {drift_data['drift_start']} "
        f"(shifted to {drift_data['shifted_to']} eV)\n"
        f"  Severity: {drift_data['severity'].upper()}"
    )
    add_user_alert(state, alert)
    
    return state


def flag_outlier(state: AgentState, sample: str, metric: str, value: float, 
                 group_mean: float, group_std: float, z_score: float) -> AgentState:
    """
    Flag statistical outlier (The Analyst function).
    
    Example:
        flag_outlier(state, "Sample_B7", "LiF_content", 65.0, 20.0, 5.0, 9.0)
        
    Agent will alert: "📊 Sample_B7 is a statistical outlier (Z-score: 9.0)"
    """
    if state['outlier_samples'] is None:
        state['outlier_samples'] = []
    
    if sample not in state['outlier_samples']:
        state['outlier_samples'].append(sample)
    
    alert = (
        f"📊 Statistical Outlier Detected!\n"
        f"  Sample: {sample}\n"
        f"  Metric: {metric}\n"
        f"  Value: {value:.1f} (Group mean: {group_mean:.1f} ± {group_std:.1f})\n"
        f"  Z-score: {z_score:.1f}\n"
        f"  Recommendation: Manual inspection advised"
    )
    add_user_alert(state, alert)
    
    return state


def initialize_state() -> AgentState:
    """
    Create fresh AgentState with all fields initialized.
    
    Call this at the start of a new session.
    """
    return AgentState(
        current_data_path=None,
        current_template_path=None,
        output_directory=None,
        current_file=None,
        data_type=None,
        recommended_workflow=None,
        triage_confidence=None,
        triage_reason=None,
        triage_parameters=None,
        quality_flags=None,
        low_quality_samples=None,
        user_quality_decision=None,
        pipeline_stage=None,
        completed_steps=[],
        pending_steps=[],
        last_analysis_summary=None,
        current_groups=None,
        group_strategy=None,
        detected_drift=None,
        outlier_samples=None,
        messages=[],
        last_tool_call=None,
        last_tool_result=None,
        user_alerts=[]
    )


# ============================================================================
# State-Aware Tool Wrapper Example
# ============================================================================

def run_xps_fit_with_state(state: AgentState, **kwargs) -> Dict[str, Any]:
    """
    Tool wrapper that automatically reads from state memory.
    
    The tool grabs paths directly from state - no need to repeat them.
    
    Example:
        User: "Fit the data"  # No need to say "in folder X"
        Agent: Calls this, which reads current_data_path from state
    """
    target_folder = state['current_data_path']
    template_path = state.get('current_template_path')
    
    if not target_folder:
        return {
            "status": "error",
            "message": "No data path in memory. Please provide folder location.",
            "user_alert": "⚠️ I don't have a data folder stored. Please tell me where the data is."
        }
    
    # Check quality flags - Gatekeeper logic
    if state.get('low_quality_samples'):
        if state.get('user_quality_decision') != "process_anyway":
            return {
                "status": "blocked",
                "message": "Quality issues detected. Awaiting user decision.",
                "user_alert": (
                    f"⚠️ Quality Check Failed!\n"
                    f"  {len(state['low_quality_samples'])} samples flagged:\n" +
                    "\n".join(f"    - {s}" for s in state['low_quality_samples'][:5]) +
                    f"\n  Should I: (1) Skip these samples, or (2) Process anyway?"
                )
            }
    
    # TODO: Call actual XPS_fitter with target_folder, template_path
    # result = xps_fitter.fit_region(target_folder, template_path, ...)
    
    # Update state with completion
    mark_step_complete(state, "xps_fitter", f"Fitted spectra")
    
    return {
        "status": "success",
        "message": f"Fitting complete on {target_folder}",
        "user_alert": None  # Or warning if issues found
    }


# ============================================================================
# LangGraph Compatible State (extends AgentState)
# ============================================================================

class GraphState(AgentState):
    """
    Extended state for LangGraph integration.
    
    Combines structured memory (AgentState) with graph-specific fields.
    """
    file_type: Optional[str]                   # 'vamas', 'txt', 'excel', 'csv'
    processing_status: Optional[str]           # 'pending', 'processing', 'complete', 'error'
    next_node: Optional[str]                   # Next graph node to execute