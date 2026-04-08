#!/usr/bin/env python3
"""
Workflow Orchestrator
=====================

Unified workflow entry point that uses LangGraph architecture:
1. Triage Node (llm_manager/triage_router.py)
2. Quality Gate Node (llm_manager/quality_gatekeeper.py)
3. Route to appropriate workflow (standard vs map)

This orchestrator sits BEFORE real_xps_workflow and routes to the correct tool chain.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Literal, List

# Add llm_manager to path
_llm_manager_path = Path(__file__).resolve().parents[1] / "llm_manager"
if str(_llm_manager_path) not in sys.path:
    sys.path.insert(0, str(_llm_manager_path))

# Import unified modules
try:
    from triage_router import (
        triage_decision_node,
        quality_gate_node,
        should_continue_processing,
        format_triage_summary
    )
    from Agentstate import AgentState, update_triage_results, add_user_alert, add_quality_flag
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: LangGraph modules not available: {e}")
    LANGGRAPH_AVAILABLE = False
    # Set stub variables
    triage_decision_node = None
    quality_gate_node = None
    should_continue_processing = None
    format_triage_summary = None
    AgentState = None
    add_user_alert = None


def create_initial_state(current_file: str, user_request: str = None, 
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Create initial AgentState for workflow orchestration.
    
    Args:
        current_file: Path to file being processed
        user_request: Optional user request/intent
        confidence_threshold: Triage confidence threshold
        
    Returns:
        Initialized AgentState dictionary
    """
    if not LANGGRAPH_AVAILABLE:
        return {"error": "LangGraph not available"}
    
    return AgentState(
        current_data_path=None,
        current_template_path=None,
        output_directory=None,
        current_file=current_file,
        data_type=None,
        recommended_workflow=None,
        triage_confidence=None,
        triage_reason=None,
        triage_parameters=None,
        quality_flags=None,
        low_quality_samples=None,
        user_quality_decision=None,
        pipeline_stage="initialization",
        completed_steps=[],
        pending_steps=[],
        error_state=None,
        user_alerts=[],
        user_request=user_request,
        confidence_threshold=confidence_threshold,
        file_path=current_file,
        quality_gate_passed=False,
        quality_report={},
        next_step="triage"
    )


def format_quality_summary(quality_report: Dict[str, Any]) -> str:
    """Create a human-readable quality summary similar to the legacy reader output."""
    if not quality_report:
        return ""

    lines: List[str] = ["=== XPS DATA QUALITY SUMMARY ==="]
    status = quality_report.get("status")
    if status:
        lines.append(f"Status: {status}")

    modality = quality_report.get("modality")
    if modality:
        lines.append(f"Modality: {modality}")

    quality_flag = quality_report.get("quality_flag")
    if quality_flag:
        lines.append(f"Overall Flag: {quality_flag}")

    snr = quality_report.get("snr")
    if isinstance(snr, (int, float)):
        lines.append(f"SNR: {snr:.2f}")

    energy_range = quality_report.get("energy_range_ev")
    if isinstance(energy_range, (int, float)):
        lines.append(f"Energy Range: {energy_range:.2f} eV")

    points = quality_report.get("data_points")
    if isinstance(points, (int, float)):
        lines.append(f"Total Data Points: {int(points)}")

    resolution = quality_report.get("resolution_pts_per_ev")
    if isinstance(resolution, (int, float)):
        lines.append(f"Resolution: {resolution:.2f} pts/eV")

    energy_shift = quality_report.get("calibration_shift_ev") or quality_report.get("energy_shift_ev")
    if isinstance(energy_shift, (int, float)):
        lines.append(f"Calibration Shift: {energy_shift:+.2f} eV")

    spatial_dims = quality_report.get("spatial_dims")
    if spatial_dims:
        lines.append(f"Spatial Dimensions: {spatial_dims}")

    recommended = quality_report.get("recommended_workflow")
    if recommended:
        lines.append(f"Recommended Workflow: {recommended}")

    critical = quality_report.get("critical_issues", [])
    warnings = quality_report.get("warnings", [])
    if critical:
        lines.append("\nCritical Issues:")
        lines.extend(f"  - {issue}" for issue in critical)
    if warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {warning}" for warning in warnings)

    lines.append("=" * 34)
    return "\n".join(lines)


def notify_quality_alerts(state: Dict[str, Any], quality_report: Dict[str, Any]) -> None:
    """Push quality alerts into AgentState for downstream display."""
    if not (LANGGRAPH_AVAILABLE and add_user_alert and quality_report):
        return

    alerts: List[str] = []
    for issue in quality_report.get("critical_issues", []):
        alerts.append(f"🚨 {issue}")
    for warning in quality_report.get("warnings", []):
        alerts.append(f"⚠️ {warning}")

    flag = quality_report.get("quality_flag")
    snr = quality_report.get("snr")
    if not alerts and flag:
        snippet = f"Quality flag: {flag}"
        if isinstance(snr, (int, float)):
            snippet += f" (SNR {snr:.2f})"
        alerts.append(f"ℹ️ {snippet}")

    for alert in alerts:
        add_user_alert(state, alert)


class WorkflowOrchestrator:
    """
    Orchestrates XPS workflow using LangGraph architecture.
    
    Flow:
        File Input → Triage → Quality Gate → Route → Execute Workflow
    """
    
    def __init__(self, confidence_threshold: float = 0.7, debug: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            confidence_threshold: Minimum confidence for triage routing
            debug: Enable debug output
        """
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("Cannot initialize orchestrator: LangGraph modules not available")
    
    def process_file(self, file_path: str, user_request: str = None) -> Dict[str, Any]:
        """
        Process XPS file through triage → quality gate → workflow routing.
        
        Args:
            file_path: Path to XPS data file
            user_request: Optional user request/intent
            
        Returns:
            Dictionary with routing decision and state
        """
        print("=" * 80)
        print("🚀 XPS WORKFLOW ORCHESTRATOR")
        print("=" * 80)
        print(f"📂 File: {Path(file_path).name}")
        print()
        
        # Step 0a: Triage Decision Node
        print("┌─────────────────────────────────────┐")
        print("│ STEP 0a: TRIAGE DECISION            │")
        print("└─────────────────────────────────────┘")
        
        state = create_initial_state(
            current_file=file_path,
            user_request=user_request,
            confidence_threshold=self.confidence_threshold
        )
        
        state = triage_decision_node(state)
        
        # Print triage results
        print(format_triage_summary(state))
        
        # Check if triage failed
        if state.get('data_type') == 'unknown':
            print("\n❌ Triage FAILED: Unknown file format")
            return {
                'success': False,
                'stage': 'triage',
                'reason': state.get('triage_reason', 'Unknown format'),
                'state': state
            }
        
        print(f"\n✅ Triage PASSED: {state.get('data_type')}")
        print()
        
        # Step 0b: Quality Gate Node
        print("┌─────────────────────────────────────┐")
        print("│ STEP 0b: QUALITY GATE               │")
        print("└─────────────────────────────────────┘")
        
        state = quality_gate_node(state)
        
        # Print quality results
        quality_report = state.get('quality_report', {})
        summary_text = format_quality_summary(quality_report)
        if summary_text:
            print(summary_text)
            notify_quality_alerts(state, quality_report)
        else:
            print("📊 Quality Status: (no report available)")
        
        # Check routing decision
        routing = should_continue_processing(state)
        
        if routing == "reject":
            print("\n❌ Quality Gate FAILED: Data quality too poor")
            return {
                'success': False,
                'stage': 'quality_gate',
                'reason': 'Critical quality issues',
                'state': state
            }
        
        if routing == "clarify":
            print("\n⚠️  Quality Gate WARNING: Poor quality, user decision required")
            return {
                'success': True,
                'stage': 'quality_gate',
                'routing': 'clarify',
                'requires_user_input': True,
                'state': state
            }
        
        print(f"\n✅ Quality Gate PASSED")
        print()
        
        # Step 0c: Determine Workflow Route
        print("┌─────────────────────────────────────┐")
        print("│ ROUTING DECISION                    │")
        print("└─────────────────────────────────────┘")
        
        recommended_workflow = state.get('recommended_workflow', 'standard_workflow')
        print(f"🔀 Routing to: {recommended_workflow}")
        
        if recommended_workflow == "map_workflow":
            print("   → Tool: XPS_mapper")
            print("   → Workflow: Map processing (clustering, MCR, visualization)")
        else:
            print("   → Tool: XPS_reader → XPS_fitter → XPS_quantifier")
            print("   → Workflow: Standard spectrum processing")
        
        print()
        print("=" * 80)
        
        return {
            'success': True,
            'stage': 'routing',
            'routing': routing,
            'workflow': recommended_workflow,
            'state': state
        }


def orchestrate_workflow(file_path: str, user_request: str = None, 
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function for workflow orchestration.
    
    Args:
        file_path: Path to XPS data file
        user_request: Optional user request
        confidence_threshold: Triage confidence threshold
        
    Returns:
        Orchestration result dictionary
    """
    orchestrator = WorkflowOrchestrator(
        confidence_threshold=confidence_threshold,
        debug=False
    )
    
    return orchestrator.process_file(file_path, user_request)


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    """Test orchestrator on example file."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="XPS Workflow Orchestrator - Triage and route XPS data files"
    )
    parser.add_argument("file_path", help="Path to XPS data file")
    parser.add_argument("--request", help="User request/intent", default=None)
    parser.add_argument("--threshold", type=float, default=0.7, 
                       help="Confidence threshold for triage")
    
    args = parser.parse_args()
    
    # Validate file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Run orchestration
    result = orchestrate_workflow(
        file_path=str(file_path),
        user_request=args.request,
        confidence_threshold=args.threshold
    )
    
    # Print result summary
    print("\n" + "=" * 80)
    print("ORCHESTRATION RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    print(f"Stage: {result['stage']}")
    
    if result['success']:
        print(f"Routing: {result.get('routing', 'N/A')}")
        print(f"Recommended Workflow: {result.get('workflow', 'N/A')}")
        
        if result.get('requires_user_input'):
            print("\n⚠️  User decision required before proceeding")
    else:
        print(f"Failure Reason: {result['reason']}")
    
    sys.exit(0 if result['success'] else 1)
