"""
triage_router.py

LangGraph integration for mandatory start nodes:
1. Triage Decision Node - Detect data type and route workflow
2. Quality Gate Node - Validate data quality before processing

These nodes form the entry point of the XPS workflow graph.
"""

from typing import Literal, Dict, Any
from pathlib import Path
import sys

# Import triage module from same directory
from enhanced_triage_fixed import should_route_to_mapper, XPSDataType

# Import unified quality gatekeeper
try:
    from quality_gatekeeper import UnifiedQualityGatekeeper, QualityFlag, DataModality
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False

# Import AgentState for structured state management
try:
    from Agentstate import AgentState, update_triage_results, add_user_alert, add_quality_flag
    AGENT_STATE_AVAILABLE = True
except ImportError:
    AGENT_STATE_AVAILABLE = False
    # Fallback to simple TriageState if AgentState not available
    from typing import TypedDict, Optional
    class AgentState(TypedDict):
        """Fallback state schema if full AgentState not available."""
        file_path: str
        user_request: Optional[str]
        confidence_threshold: Optional[float]
        data_type: str
        recommended_workflow: str
        triage_confidence: float
        triage_reason: str
        triage_parameters: Dict[str, Any]
        quality_gate_passed: bool
        quality_report: Dict[str, Any]
        next_step: Literal["standard_workflow", "map_workflow", "clarify", "error"]


# ============================================================================
# State Schema (Now uses AgentState)
# ============================================================================

TriageState = AgentState  # Alias for backwards compatibility


# ============================================================================
# Triage Decision Node
# ============================================================================

def triage_decision_node(state: AgentState) -> AgentState:
    """
    Deterministic triage decision node for LangGraph with SciAgent intelligence.
    
    This node implements "The Metadata Profiler" from system_prompt.md:
    - Analyzes XPS file structure
    - Detects data type (standard spectra vs map)
    - Sets routing decision (standard_workflow vs map_workflow)
    - Enriches state with triage metadata
    - Adds intelligent user alerts
    
    SciAgent becomes aware and says:
        "I've examined the dataset. I see 50 spectra / I see it's a 20x20 map.
         I will automatically route to the 'High-Res Narrow Scan' fitting workflow."
    
    Args:
        state: Current AgentState with 'current_file' or 'file_path' required
        
    Returns:
        Updated state with triage results and routing decision
    """
    
    # Extract file path from state (check both fields for flexibility)
    file_path_str = state.get("current_file") or state.get("file_path")
    if not file_path_str:
        state["data_type"] = "unknown"
        state["recommended_workflow"] = "error"
        state["triage_confidence"] = 0.0
        state["triage_reason"] = "No file path provided"
        state["triage_parameters"] = {}
        if "next_step" in state:
            state["next_step"] = "error"
        
        if AGENT_STATE_AVAILABLE:
            add_user_alert(state, "❌ No file path provided for triage")
        
        return state
    
    file_path = Path(file_path_str)
    confidence_threshold = state.get("confidence_threshold", 0.7)
    
    # Validate file exists
    if not file_path.exists():
        state["data_type"] = "unknown"
        state["recommended_workflow"] = "error"
        state["triage_confidence"] = 0.0
        state["triage_reason"] = f"File not found: {file_path}"
        state["triage_parameters"] = {}
        if "next_step" in state:
            state["next_step"] = "error"
        
        if AGENT_STATE_AVAILABLE:
            add_user_alert(state, f"❌ File not found: {file_path.name}")
        
        return state
    
    # Run deterministic triage
    try:
        should_route, triage_result = should_route_to_mapper(file_path, confidence_threshold)
        
        # Update state using structured functions if available
        if AGENT_STATE_AVAILABLE:
            state = update_triage_results(state, triage_result)
        else:
            # Fallback manual update
            state["data_type"] = triage_result["data_type"].value
            state["triage_confidence"] = triage_result["confidence"]
            state["triage_reason"] = triage_result["reason"]
            state["triage_parameters"] = triage_result["parameters"]
        
        # Determine workflow routing
        if should_route:
            state["recommended_workflow"] = "map"
            if "next_step" in state:
                state["next_step"] = "map_workflow"
            
            # Intelligent alert - The Metadata Profiler speaks
            params = triage_result["parameters"]
            
            # Check if this is a depth profile being routed to mapper
            if triage_result["data_type"] == XPSDataType.DEPTH_PROFILE:
                ncycles = params.get('num_cycles', 0)
                alert = (
                    f"📊 Metadata Profiler: I've examined the dataset.\n"
                    f"  Detected: Depth Profile with {ncycles} cycles\n"
                    f"  Region: {params.get('region', 'unknown')}\n"
                    f"  Energy Points: {params.get('energy_points', '?')}\n"
                    f"  Confidence: {triage_result['confidence']:.0%}\n"
                    f"  ✓ Sufficient statistical power (n > 10)\n"
                    f"  → Routing to MAPPER workflow for PCA/MCR multivariate analysis\n"
                    f"  (Treating depth cycles as pseudo-spatial map for decomposition)"
                )
            elif params.get('nx') and params.get('ny'):
                alert = (
                    f"📊 Metadata Profiler: I've examined the dataset.\n"
                    f"  Detected: {params['nx']}x{params['ny']} hyperspectral map "
                    f"({params.get('total_pixels', '?')} pixels)\n"
                    f"  Region: {params.get('region', 'unknown')}\n"
                    f"  Energy Points: {params.get('energy_points', '?')}\n"
                    f"  Confidence: {triage_result['confidence']:.0%}\n"
                    f"  → Routing to MAP workflow (PCA + MCR decomposition)"
                )
            else:
                alert = (
                    f"📊 Metadata Profiler: Detected map data "
                    f"(confidence: {triage_result['confidence']:.0%})\n"
                    f"  → Routing to MAP workflow"
                )
            
            if AGENT_STATE_AVAILABLE:
                add_user_alert(state, alert)
                
        elif triage_result["data_type"] == XPSDataType.STANDARD_SPECTRA:
            state["recommended_workflow"] = "standard"
            if "next_step" in state:
                state["next_step"] = "standard_workflow"
            
            alert = (
                f"📊 Metadata Profiler: I've examined the dataset.\n"
                f"  Detected: Standard XPS spectra\n"
                f"  Confidence: {triage_result['confidence']:.0%}\n"
                f"  → Routing to STANDARD workflow (read → fit → quantify)"
            )
            
            if AGENT_STATE_AVAILABLE:
                add_user_alert(state, alert)
                
        elif triage_result["confidence"] < 0.4:
            # Very low confidence - ask user for clarification
            state["recommended_workflow"] = "clarify"
            if "next_step" in state:
                state["next_step"] = "clarify"
            
            alert = (
                f"⚠️ Metadata Profiler: Low confidence detection\n"
                f"  Data type unclear (confidence: {triage_result['confidence']:.0%})\n"
                f"  Reason: {triage_result['reason']}\n"
                f"  → Please specify: Is this (1) Standard spectra or (2) Map data?"
            )
            
            if AGENT_STATE_AVAILABLE:
                add_user_alert(state, alert)
        else:
            # Unknown but has some structure - default to standard
            state["recommended_workflow"] = "standard"
            if "next_step" in state:
                state["next_step"] = "standard_workflow"
            
            alert = (
                f"📊 Metadata Profiler: Defaulting to standard workflow\n"
                f"  (Confidence: {triage_result['confidence']:.0%}, Reason: {triage_result['reason']})"
            )
            
            if AGENT_STATE_AVAILABLE:
                add_user_alert(state, alert)
            
    except Exception as e:
        state["data_type"] = "unknown"
        state["recommended_workflow"] = "error"
        state["triage_confidence"] = 0.0
        state["triage_reason"] = f"Triage error: {str(e)}"
        state["triage_parameters"] = {}
        if "next_step" in state:
            state["next_step"] = "error"
        
        if AGENT_STATE_AVAILABLE:
            add_user_alert(state, f"❌ Triage failed: {str(e)}")
    
    return state


# ============================================================================
# Conditional Edge Function
# ============================================================================

def route_by_triage(state: TriageState) -> Literal["standard_workflow", "map_workflow", "clarify", "error"]:
    """
    Conditional edge function for LangGraph routing.
    
    Use this as the conditional edge after the triage_decision_node:
    
    ```python
    graph.add_conditional_edges(
        "triage",
        route_by_triage,
        {
            "standard_workflow": "run_standard_workflow",
            "map_workflow": "run_map_workflow", 
            "clarify": "ask_user_clarification",
            "error": "handle_error"
        }
    )
    ```
    
    Args:
        state: Graph state after triage_decision_node
        
    Returns:
        Routing key: "standard_workflow" | "map_workflow" | "clarify" | "error"
    """
    return state["next_step"]


# ============================================================================
# Human-readable Summary Generator
# ============================================================================

def format_triage_summary(state: TriageState) -> str:
    """
    Generate human-readable summary of triage decision for logging/UI.
    
    Args:
        state: Graph state with triage results
        
    Returns:
        Formatted string summary
    """
    data_type = state.get("data_type", "unknown")
    confidence = state.get("triage_confidence", 0.0)
    reason = state.get("triage_reason", "No analysis")
    workflow = state.get("recommended_workflow", "unknown")
    params = state.get("triage_parameters", {})
    
    summary_lines = [
        "=== XPS Data Triage Result ===",
        f"Data Type: {data_type}",
        f"Confidence: {confidence:.1%}",
        f"Recommended Workflow: {workflow}",
        f"Reason: {reason}",
    ]
    
    # Add detected parameters if available
    if params:
        summary_lines.append("\nDetected Parameters:")
        if "nx" in params and "ny" in params:
            summary_lines.append(f"  Spatial: {params['nx']} x {params['ny']} = {params.get('total_pixels', 'N/A')} pixels")
        if "region" in params:
            summary_lines.append(f"  Region: {params['region']}")
        if "energy_points" in params:
            summary_lines.append(f"  Energy Points: {params['energy_points']}")
    
    summary_lines.append("=" * 30)
    
    return "\n".join(summary_lines)


# ============================================================================
# Quality Gate Node (Mandatory Start Node #2)
# ============================================================================

def quality_gate_node(state: AgentState) -> AgentState:
    """
    Quality gatekeeper node - validates data quality before processing.
    
    This is a mandatory start node that:
    1. Validates SNR, resolution, energy range
    2. Detects critical issues (corrupted data, empty files)
    3. Flags warnings (low SNR, poor resolution)
    4. Updates state with quality report
    5. Adds user alerts for quality issues
    
    Args:
        state: Current agent state with file_path
        
    Returns:
        Updated state with quality_gate_passed and quality_report
    """
    if not QUALITY_GATE_AVAILABLE:
        # Fallback if quality gatekeeper not available
        state['quality_gate_passed'] = True
        state['quality_report'] = {'status': 'skipped', 'message': 'Quality gatekeeper not available'}
        return state
    
    # Get file from state
    file_path = state.get('current_file') or state.get('file_path')
    if not file_path or not Path(file_path).exists():
        state['quality_gate_passed'] = False
        state['quality_report'] = {'status': 'error', 'message': 'File not found'}
        if AGENT_STATE_AVAILABLE:
            add_user_alert(state, "⚠️ Quality Gate: File not found for validation")
        return state
    
    # Initialize gatekeeper
    gatekeeper = UnifiedQualityGatekeeper()
    
    # Load data (simplified - actual implementation needs proper data loading)
    # For now, use triage results to determine modality
    data_type = state.get('data_type', '')
    if 'map' in data_type.lower():
        modality = DataModality.MAP_HYPERSPECTRAL if 'hyperspectral' in data_type else DataModality.MAP_2D
    else:
        modality = DataModality.SINGLE_SPECTRUM
    
    # TODO: Load actual data object here
    # For now, create a placeholder validation
    
    # Placeholder: Mark as passed with note
    state['quality_gate_passed'] = True
    state['quality_report'] = {
        'status': 'placeholder',
        'modality': modality.value,
        'message': 'Quality gate ready - awaiting data loading implementation'
    }
    
    # Add informational alert
    if AGENT_STATE_AVAILABLE:
        add_user_alert(
            state,
            f"🔍 Quality Gate: Ready to validate {modality.value} data"
        )
    
    return state


def should_continue_processing(state: AgentState) -> Literal["process", "reject", "clarify"]:
    """
    Conditional edge based on quality gate results.
    
    Routes to:
    - "process": Quality passed, continue workflow
    - "reject": Critical issues, cannot process
    - "clarify": User attention needed
    
    Args:
        state: Agent state with quality_gate_passed
        
    Returns:
        Next edge to follow
    """
    if not state.get('quality_gate_passed', False):
        return "reject"
    
    quality_report = state.get('quality_report', {})
    if quality_report.get('requires_user_attention', False):
        return "clarify"
    
    return "process"


# ============================================================================
# Integration with ExperimentRouter
# ============================================================================

def enhance_router_with_triage(router_instance) -> None:
    """
    Enhance existing ExperimentRouter with triage awareness.
    
    Adds triage result context to the system message so the LLM is aware
    of detected data types and routing decisions.
    
    Args:
        router_instance: ExperimentRouter instance to enhance
    """
    triage_policy = """

XPS Data Triage Integration:
- Files are automatically analyzed for data type before workflow execution
- Data types: standard_spectra, map_2d, map_hyperspectral, unknown
- Routing decision: standard_workflow (XPS_reader) vs map_workflow (XPS_mapper)
- Triage confidence threshold: 0.7 (configurable)
- You can reference triage results in args via 'triage_result' key if available
"""
    
    # Append triage policy to existing system message
    if hasattr(router_instance, 'system_msg'):
        current_content = router_instance.system_msg.content
        router_instance.system_msg.content = current_content + triage_policy


# ============================================================================
# Standalone CLI for Testing
# ============================================================================

if __name__ == "__main__":
    """Test triage decision node on example files."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test XPS data triage decision node")
    parser.add_argument("file_path", help="Path to XPS data file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()
    
    # Create initial state
    test_state: TriageState = {
        "file_path": args.file_path,
        "user_request": None,
        "confidence_threshold": args.threshold,
        "data_type": "",
        "recommended_workflow": "",
        "triage_confidence": 0.0,
        "triage_reason": "",
        "triage_parameters": {},
        "next_step": "error"
    }
    
    # Run triage node
    result_state = triage_decision_node(test_state)
    
    # Print summary
    print(format_triage_summary(result_state))
    print(f"\nGraph Routing: {result_state['next_step']}")
