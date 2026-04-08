import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Define colors
color_start = '#e3f2fd'  # Light blue
color_process = '#fff3e0'  # Light orange
color_decision = '#fff9c4'  # Light yellow
color_analysis = '#e8f5e9'  # Light green
color_output = '#f3e5f5'  # Light purple
color_validation = '#ffebee'  # Light red

# Box style parameters
box_width = 2.5
box_height = 0.8
decision_width = 2.2
decision_height = 1.0
rounded = 0.1
fontsize = 10
title_fontsize = 14

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, style='round'):
    if style == 'round':
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle=f"round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
    elif style == 'diamond':
        # Create diamond shape for decision boxes
        diamond = mpatches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                         boxstyle="round,pad=0.1",
                                         facecolor=color, edgecolor='black', linewidth=1.5,
                                         transform=ax.transData)
    ax.add_patch(box if style != 'diamond' else diamond)
    
    # Add text
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            weight='normal', wrap=True)
    
    return box

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, text='', curved=False):
    if curved:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               connectionstyle="arc3,rad=0.3",
                               arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='black')
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='black')
    ax.add_patch(arrow)
    
    if text:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, text, fontsize=9, style='italic')

# Title
ax.text(5, 19, 'MCR-ALS Workflow for XPS Data Analysis', 
        ha='center', va='center', fontsize=title_fontsize, weight='bold')

# Phase 1: Data Preparation
y_pos = 17.5
create_box(ax, 5, y_pos, box_width, box_height, 
          'Raw XPS Data\n(Survey + Narrow Scans)', color_start)

# Information gathering
y_pos -= 1.5
create_box(ax, 2, y_pos, box_width*0.9, box_height*1.2,
          'Gather Information:\n• Research question\n• Expected chemistry\n• Complementary data',
          color_process)

# Data visualization
create_box(ax, 5, y_pos, box_width*0.9, box_height*1.2,
          'Visualize Data:\n• Overlay plots\n• Waterfall plots\n• Check for changes',
          color_process)

# Outlier detection
create_box(ax, 8, y_pos, box_width*0.9, box_height*1.2,
          'Initial Screening:\n• PRE analysis\n• PCA outliers\n• Trajectories',
          color_process)

# Arrows from raw data
create_arrow(ax, 5, 17.1, 2, y_pos+0.6)
create_arrow(ax, 5, 17.1, 5, y_pos+0.6)
create_arrow(ax, 5, 17.1, 8, y_pos+0.6)

# Decision: Preprocessing
y_pos -= 2
create_box(ax, 5, y_pos, decision_width*1.2, decision_height,
          'Preprocessing\nNeeded?', color_decision)

# Connect exploration boxes to decision
create_arrow(ax, 2, 15.5, 3.5, y_pos+0.5)
create_arrow(ax, 5, 15.5, 5, y_pos+0.5)
create_arrow(ax, 8, 15.5, 6.5, y_pos+0.5)

# Preprocessing options
y_pos -= 1.8
create_box(ax, 2.5, y_pos, box_width*0.8, box_height*1.5,
          'Preprocessing:\n• Normalize\n• Charge correct\n• Concatenate\n• Variable select',
          color_process)

# No preprocessing path
ax.text(7.5, y_pos, 'Minimal/None\n(Preferred)', fontsize=9, style='italic', ha='center')

# Arrows from decision
create_arrow(ax, 4, 13.5, 2.5, y_pos+0.75, text='Yes')
create_arrow(ax, 6, 13.5, 7.5, y_pos+0.5, text='No', curved=True)

# Phase 2: MCR Analysis
y_pos -= 2
create_box(ax, 5, y_pos, box_width*1.1, box_height,
          'MCR-ALS Analysis', color_analysis)

# Connect preprocessing paths
create_arrow(ax, 2.5, 11.2, 4, y_pos+0.4)
create_arrow(ax, 7.5, 11.2, 6, y_pos+0.4)

# Factor selection
y_pos -= 1.5
create_box(ax, 5, y_pos, box_width*1.2, box_height*1.3,
          'Factor Selection:\n• Scree plot\n• Cross-validation\n• Reconstruction test',
          color_process)

create_arrow(ax, 5, 10.2, 5, y_pos+0.65)

# Decision: Factors sensible?
y_pos -= 1.8
create_box(ax, 5, y_pos, decision_width*1.3, decision_height,
          'Factors\nChemically\nSensible?', color_decision)

create_arrow(ax, 5, 8.35, 5, y_pos+0.5)

# Iteration arrow
create_arrow(ax, 6.5, y_pos, 7.5, 8.7, curved=True)
ax.text(8.2, 7.5, 'Adjust\nfactors', fontsize=9, style='italic')

# Phase 3: Analysis & Interpretation
y_pos -= 1.8
create_box(ax, 5, y_pos, box_width, box_height*1.2,
          'Peak Fit\nMCR Factors', color_analysis)

create_arrow(ax, 5, 6.7, 5, y_pos+0.6, text='Yes')

# Parallel analyses
y_pos -= 1.5
create_box(ax, 2.5, y_pos, box_width*0.9, box_height*1.2,
          'Identify\nIntermediates\n(Conc. profiles)',
          color_analysis)

create_box(ax, 5, y_pos, box_width*0.9, box_height*1.2,
          'Cluster Analysis\n(Hierarchical)',
          color_analysis)

create_box(ax, 7.5, y_pos, box_width*0.9, box_height*1.2,
          'Chemical\nQuantification',
          color_analysis)

# Connect peak fitting to analyses
create_arrow(ax, 4.2, 5.1, 2.5, y_pos+0.6)
create_arrow(ax, 5, 5.1, 5, y_pos+0.6)
create_arrow(ax, 5.8, 5.1, 7.5, y_pos+0.6)

# Phase 4: Validation
y_pos -= 2
create_box(ax, 5, y_pos, decision_width*1.5, decision_height*1.1,
          'Validate Against\nRaw Data?', color_validation)

# Connect analyses to validation
create_arrow(ax, 2.5, 3.5, 3.8, y_pos+0.5)
create_arrow(ax, 5, 3.5, 5, y_pos+0.55)
create_arrow(ax, 7.5, 3.5, 6.2, y_pos+0.5)

# Iteration back to MCR
create_arrow(ax, 3.5, y_pos, 1, 10.5, curved=True)
ax.text(0.5, 6, 'Issues\nfound', fontsize=9, style='italic', rotation=90)

# Final output
y_pos -= 1.5
create_box(ax, 5, y_pos, box_width*1.3, box_height*1.2,
          'Report Results:\n• Pure spectra\n• Concentrations\n• Chemical states',
          color_output)

create_arrow(ax, 5, 2.5, 5, y_pos+0.6, text='Validated')

# Add legend
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_start, edgecolor='black', label='Input Data'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_process, edgecolor='black', label='Processing Step'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_decision, edgecolor='black', label='Decision Point'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_analysis, edgecolor='black', label='Analysis'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_validation, edgecolor='black', label='Validation'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_output, edgecolor='black', label='Output')
]

ax.legend(handles=legend_elements, loc='lower center', ncol=3, 
         bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=10)

# Add phase labels on the side
phase_x = 9.5
ax.text(phase_x, 16.5, 'Phase 1:\nData\nPreparation', fontsize=11, 
        weight='bold', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

ax.text(phase_x, 10, 'Phase 2:\nMCR-ALS\nAnalysis', fontsize=11,
        weight='bold', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

ax.text(phase_x, 4.5, 'Phase 3:\nInterpretation', fontsize=11,
        weight='bold', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

ax.text(phase_x, 1.5, 'Phase 4:\nValidation &\nReporting', fontsize=11,
        weight='bold', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

plt.tight_layout()
plt.savefig('mcr_als_workflow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('mcr_als_workflow.png', dpi=300, bbox_inches='tight')
plt.show()
