"""
xps_workflow_graph.py

Example LangGraph workflow with integrated XPS data triage decision node.
Demonstrates branching between standard and map workflows based on deterministic triage.
"""

from typing import TypedDict, Literal
from pathlib import Path

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: langgraph not installed. Install with: pip install langgraph")

# Local imports
from triage_router import (
    TriageState, 
    triage_decision_node, 
    route_by_triage,
    format_triage_summary
)


# ============================================================================
# Workflow Implementation Nodes
# ============================================================================

def run_standard_workflow(state: TriageState) -> TriageState:
    """
    Execute standard XPS workflow (XPS_reader path).
    
    This node would call your existing standard XPS processing:
    - Read spectra with XPS_reader
    - Peak fitting
    - Quantification
    - Standard plots
    """
    print(f"\n🔬 Running STANDARD workflow for: {Path(state['file_path']).name}")
    print(format_triage_summary(state))
    
    # TODO: Call actual standard workflow
    # from Tools.XPS_reader import xps_reader
    # result = xps_reader.process_file(state['file_path'])
    
    state["workflow_status"] = "standard_completed"
    state["workflow_result"] = {
        "type": "standard",
        "regions_processed": state.get("triage_parameters", {}).get("regions_detected", [])
    }
    
    return state


def run_map_workflow(state: TriageState) -> TriageState:
    """
    Execute map/hyperspectral workflow (XPS_mapper path).
    
    This node would call your XPS_mapper processing:
    - Parse hyperspectral map data
    - PCA + clustering
    - MCR decomposition
    - Quality metrics + chemical ID
    - Map visualizations
    """
    print(f"\n🗺️  Running MAP workflow for: {Path(state['file_path']).name}")
    print(format_triage_summary(state))
    
    params = state.get("triage_parameters", {})
    print(f"\nMap dimensions: {params.get('nx', '?')} x {params.get('ny', '?')} pixels")
    print(f"Region: {params.get('region', 'unknown')}")
    print(f"Energy points: {params.get('energy_points', '?')}")
    
    # TODO: Call actual map workflow
    # from Tools.XPS_mapper import XPS_map
    # result = XPS_map.process_hyperspectral_map(state['file_path'], ...)
    
    state["workflow_status"] = "map_completed"
    state["workflow_result"] = {
        "type": "map",
        "dimensions": (params.get("nx"), params.get("ny")),
        "region": params.get("region")
    }
    
    return state


def ask_user_clarification(state: TriageState) -> TriageState:
    """
    Request user clarification when triage confidence is low.
    
    This node would interact with user to determine correct workflow.
    """
    print(f"\n❓ Need clarification for: {Path(state['file_path']).name}")
    print(f"   Triage confidence: {state['triage_confidence']:.1%}")
    print(f"   Reason: {state['triage_reason']}")
    print("\n   Please specify: 'standard' or 'map' workflow")
    
    # TODO: Actual user interaction
    # user_choice = get_user_input()
    user_choice = "standard"  # Placeholder
    
    if user_choice == "map":
        state["next_step"] = "map_workflow"
    else:
        state["next_step"] = "standard_workflow"
    
    state["workflow_status"] = "clarified"
    
    return state


def handle_error(state: TriageState) -> TriageState:
    """Handle triage or workflow errors."""
    print(f"\n❌ Error: {state['triage_reason']}")
    state["workflow_status"] = "error"
    return state


# ============================================================================
# Build LangGraph Workflow
# ============================================================================

def build_xps_workflow_graph() -> StateGraph:
    """
    Build LangGraph workflow with triage-based routing.
    
    Graph structure:
        START
          ↓
        [triage_decision_node] ← Deterministic Python triage
          ↓
        (route_by_triage) ← Conditional edge
          ├→ "standard_workflow" → [run_standard_workflow] → END
          ├→ "map_workflow" → [run_map_workflow] → END
          ├→ "clarify" → [ask_user_clarification] → (re-route)
          └→ "error" → [handle_error] → END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph not installed")
    
    # Initialize graph with TriageState schema
    workflow = StateGraph(TriageState)
    
    # Add nodes
    workflow.add_node("triage", triage_decision_node)
    workflow.add_node("run_standard_workflow", run_standard_workflow)
    workflow.add_node("run_map_workflow", run_map_workflow)
    workflow.add_node("clarify", ask_user_clarification)
    workflow.add_node("error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("triage")
    
    # Add conditional routing after triage
    workflow.add_conditional_edges(
        "triage",
        route_by_triage,
        {
            "standard_workflow": "run_standard_workflow",
            "map_workflow": "run_map_workflow",
            "clarify": "clarify",
            "error": "error"
        }
    )
    
    # Add edges to END
    workflow.add_edge("run_standard_workflow", END)
    workflow.add_edge("run_map_workflow", END)
    workflow.add_edge("error", END)
    
    # Clarification re-routes based on user decision
    workflow.add_conditional_edges(
        "clarify",
        lambda state: state["next_step"],
        {
            "standard_workflow": "run_standard_workflow",
            "map_workflow": "run_map_workflow"
        }
    )
    
    return workflow.compile()


# ============================================================================
# Execution Helper
# ============================================================================

def run_xps_file_analysis(file_path: str, confidence_threshold: float = 0.7):
    """
    Execute XPS analysis workflow with automatic triage-based routing.
    
    Args:
        file_path: Path to XPS data file
        confidence_threshold: Triage confidence threshold (default 0.7)
        
    Returns:
        Final state after workflow execution
    """
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available - falling back to direct triage")
        from triage_router import triage_decision_node
        initial_state = {
            "file_path": file_path,
            "confidence_threshold": confidence_threshold,
            "user_request": None,
            "data_type": "",
            "recommended_workflow": "",
            "triage_confidence": 0.0,
            "triage_reason": "",
            "triage_parameters": {},
            "next_step": "error"
        }
        result = triage_decision_node(initial_state)
        print(format_triage_summary(result))
        return result
    
    # Build and run graph
    graph = build_xps_workflow_graph()
    
    initial_state = {
        "file_path": file_path,
        "confidence_threshold": confidence_threshold,
        "user_request": None,
        "data_type": "",
        "recommended_workflow": "",
        "triage_confidence": 0.0,
        "triage_reason": "",
        "triage_parameters": {},
        "next_step": "error"
    }
    
    # Execute graph
    print(f"\n{'='*60}")
    print(f"Starting XPS Workflow Graph")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    final_state = graph.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"Workflow Complete: {final_state.get('workflow_status', 'unknown')}")
    print(f"{'='*60}\n")
    
    return final_state


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    """
    Test the workflow graph on example XPS files.
    
    Usage:
        python xps_workflow_graph.py path/to/file.csv
        python xps_workflow_graph.py path/to/file.csv --threshold 0.6
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="XPS Workflow Graph with Triage-based Routing"
    )
    parser.add_argument("file_path", help="Path to XPS data file")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7,
        help="Triage confidence threshold (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        exit(1)
    
    # Run workflow
    final_state = run_xps_file_analysis(args.file_path, args.threshold)
    
    # Print final results
    print("\nFinal State:")
    print(f"  Data Type: {final_state.get('data_type')}")
    print(f"  Workflow: {final_state.get('recommended_workflow')}")
    print(f"  Status: {final_state.get('workflow_status')}")
    if final_state.get('workflow_result'):
        print(f"  Result: {final_state['workflow_result']}")
